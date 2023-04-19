from collections import namedtuple
import math
import torch.optim as optim
import torch
import torch.nn as nn
from train import finetune, eval_model
from quant_utils import *
from model import *
from enum import Enum
import numpy as np
import args
from data import get_cifar10_dataloader
from copy import deepcopy


State = namedtuple('State', 'method, idx, num_heads, in_dim, out_dim, prec')


class Stage(Enum):
    Quant = 0
    Prune = 1


class Strategy:
    def __init__(self, target_len: int, val) -> None:
        self.target_len = target_len
        self.cur_strategy = []
        self.fill(val)

    @property
    def strategy(self):
        assert len(self) == self.target_len, f"len: {len(self)}, target: {self.target_len}, val: {self.cur_strategy}"
        return self.cur_strategy

    def __len__(self):
        return len(self.cur_strategy)

    def set(self, idx, val):
        self.cur_strategy[idx] = val

    @property
    def is_full(self):
        return len(self) == self.cur_strategy

    def clear(self):
        self.cur_strategy = []

    def fill(self, val):
        self.cur_strategy += [val] * (self.target_len-len(self))
        return self

class Env:
    def __init__(self, model: CAFIA_Transformer, weight_path, trainloader, testloader, device, args) -> None:
        self.model = model
        self.weight_path = weight_path
        self.load_weight()

        self.trainloader = trainloader
        self.testloader = testloader

        # resouce bound
        self.lat_b = args.lat_b
        self.e_b = args.e_b
        self.mem_b = args.mem_b

        # quant bit range
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.a_bit = args.a_bit

        # prune heads
        self.max_heads = args.max_heads
        self.min_heads = args.min_heads
        self.head_dim = args.head_dim

        self.device = device
        self.ori_acc = args.ori_acc
        self.float_bit = 8

        self._build_state()
        # runtime
        self.cur_idx = 0
        self.best_reward = -math.inf
        self.quant_strategy = Strategy(len(self.quant_idx), self.max_bit)
        self.prune_strategy = Strategy(len(self.prune_idx), self.max_heads)
        self.stage = Stage.Quant

    def _apply_strategy(self):
        # self._apply_quant()
        self.quant_strategy.strategy
        self.prune_strategy.strategy

    def _apply_quant(self):
        set_mixed_precision(self.model, self.quant_idx, mix_weight_act_strategy(
            self.quant_strategy.strategy, [self.a_bit] * len(self.quant_strategy)))

    def step(self, action):
        action = self._action_wall(action)

        if self.stage is Stage.Quant:
            self.quant_strategy.set(self.cur_idx, action)
        else:
            self.prune_strategy.set(self.cur_idx, action)

        info_set = {'info': f"{self.stage.name} take action {action}"}

        if self._is_batch_end():
            self._adjust_strategy()

            self._update_states()
            self._apply_strategy()
            # reward = self.reward()
            reward = self.test_reward()
            done = True

            info_set['info'] = f'{self.stage.name} finish\nquant_pi:{self.quant_strategy.strategy}\nprune_pi:{self.prune_strategy.strategy}\nreward: {reward}\n'

            next_state = self.reset()
            return next_state, reward, done, info_set

        self.cur_idx += 1
        next_state = self._get_next_states()
        reward = 0
        done = False

        return next_state, reward, done, info_set

    def show_state(self, norm=True):
        S = []
        if norm:
            S += [self.prune_states, self.quant_states]
        else:
            S += [self._prune_states, self._quant_states]

        s = ''
        for states in S:
            for i in range(states.shape[0]):
                # method, *rest = state
                # s += f'{Stage.Quant if int(method)==1 else Stage.Prune},'
                # for r in rest:
                #     s += f'{int(r)}'
                for d in states[i]:
                    s += f'{int(d)}, ' if not norm else f'{d}, '
                s += '\n'
            s += '==========================================\n'
        print(s)

    def _update_states(self):
        # method, idx, num_heads, in_dim, out_dim, prec1, prec2
        # 0        1    2           3      4        5      6

        #  0     1     2     3
        #  l --  l --  l --  l
        # msa - msa - mlp -mlp
        # update states according to strategy
        for idx, b in enumerate(self.quant_strategy.strategy):
            self._quant_states[idx][-1] = b
            self._quant_states[idx][-2] = b
            # update prune state
            fc_idx = idx % 4
            if fc_idx in [0, 1]:
                msa_idx = idx // 4
                self._prune_states[msa_idx][fc_idx-2] = b

        for idx, heads in enumerate(self.prune_strategy.strategy):
            # update prune state
            self._prune_states[idx][2] = heads
            # update quant state
            fc_idxs = [idx*4, idx*4+1]
            self._quant_states[fc_idxs[0]][4] = self.head_dim * heads
            self._quant_states[fc_idxs[1]][3] = self.head_dim * heads

    def _get_next_states(self, norm=False):
        if self.stage is Stage.Quant:
            return self.quant_states[self.cur_idx, :].copy() if norm else self._quant_states[self.cur_idx, :].copy()
        return self.prune_states[self.cur_idx, :].copy() if norm else self._prune_states[self.cur_idx, :].copy()

    def test_reward(self):
        s = -abs(5 - sum(self.quant_strategy.strategy)/len(self.quant_strategy))
        return s

    def reward(self):
        acc = eval_model(finetune(self.model, self.trainloader,
                         self.device), self.testloader, self.device)
        return (acc - self.ori_acc) * 0.1

    def _adjust_strategy(self):
        pass

    def reset(self):
        self.load_weight()
        self.cur_idx = 0
        self.stage = Stage.Quant if self.stage is Stage.Prune else Stage.Prune
        return self._get_next_states()

    def _action_wall(self, action):
        action = float(action)
        if self.stage is Stage.Quant:
            lbound, rbound = self.min_bit - 0.5, self.max_bit + \
                0.5  # same stride length for each bit
        else:
            lbound, rbound = self.min_heads - 0.5, self.max_heads + 0.5
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action

    def load_weight(self):
        load_weight_for_vit(self.model, self.weight_path)

    def _is_batch_end(self):
        if self.stage is Stage.Quant:
            return self.cur_idx == len(self.quant_idx)-1
        if self.stage is Stage.Prune:
            return self.cur_idx == len(self.prune_idx)-1

    def _build_state(self):
        prune_idx = []
        quant_idx = []

        prune_states = []
        quant_states = []
        i, j = 0, 0

        for idx, m in enumerate(self.model.modules()):
            if type(m) is SelfAttention:
                this_state = []
                out_dim = in_dim = m.head_dim*m.heads
                this_state.append([Stage.Prune.value])
                this_state.append([i])
                this_state.append([m.heads])
                this_state.append([in_dim])
                this_state.append([out_dim])
                this_state.append([self.float_bit])  # prec1
                this_state.append([self.float_bit])  # prec2
                prune_states.append(np.hstack(this_state))
                i += 1
                prune_idx.append(idx)

                for _ in range(2):
                    this_state = []
                    this_state.append([Stage.Quant.value])
                    this_state.append([j])
                    this_state.append([1])
                    this_state.append([in_dim])
                    this_state.append([out_dim])
                    this_state.append([self.float_bit])  # prec1
                    this_state.append([self.float_bit])  # prec2
                    quant_states.append(np.hstack(this_state))
                    j += 1
                    quant_idx.append(idx)
            if type(m) is QLinear:
                in_dim = m.in_features
                out_dim = m.out_features
                this_state = []
                this_state.append([Stage.Quant.value])
                this_state.append([j])
                this_state.append([1])
                this_state.append([in_dim])
                this_state.append([out_dim])
                this_state.append([self.float_bit])  # prec1
                this_state.append([self.float_bit])  # prec2
                quant_states.append(np.hstack(this_state))
                j += 1
                quant_idx.append(idx)

        self._quant_states = np.array(quant_states, 'float')
        self._prune_states = np.array(prune_states, 'float')

        print(
            '=> shape of quant state (n_layer * n_dim): {}'.format(self._quant_states.shape))
        print(
            '=> shape of prune state (n_layer * n_dim): {}'.format(self._prune_states.shape))

        # quant state scale
        _qmin = [1, 0, 1]
        _qmin.append(
            min(min(self._quant_states[:, 3]), self.min_heads*self.head_dim))
        _qmin.append(
            min(min(self._quant_states[:, 4]), self.min_heads*self.head_dim))
        _qmin.append(self.min_bit)
        _qmin.append(self.min_bit)

        _qmax = [1, max(self._quant_states[:, 1]), 1]
        _qmax.append(
            max(max(self._quant_states[:, 3]), self.max_heads*self.head_dim))
        _qmax.append(
            max(max(self._quant_states[:, 4]), self.max_heads*self.head_dim))
        _qmax.append(self.max_bit)
        _qmax.append(self.max_bit)

        # prune state scale
        _pmin = [1, 0, self.min_heads]
        _pmin.append(
            min(min(self._prune_states[:, 3]), self.min_heads*self.head_dim))
        _pmin.append(
            min(min(self._prune_states[:, 4]), self.min_heads*self.head_dim))
        _pmin.append(self.min_bit)
        _pmin.append(self.min_bit)

        _pmax = [1, max(self._prune_states[:, 1]), self.max_heads]
        _pmax.append(
            max(max(self._prune_states[:, 3]), self.max_heads*self.head_dim))
        _pmax.append(
            max(max(self._prune_states[:, 4]), self.max_heads*self.head_dim))
        _pmax.append(self.max_bit)
        _pmax.append(self.max_bit)

        self._qmax, self._qmin, self._pmax, self._pmin = _qmax, _qmin, _pmax, _pmin

        self.quant_idx = quant_idx
        self.prune_idx = prune_idx

    @property
    def quant_states(self):
        states = self._quant_states.copy()
        for i in range(states.shape[1]):
            fmin = self._qmin[i]
            fmax = self._qmax[i]
            if fmax - fmin > 0:
                states[:, i] = (states[:, i] - fmin) / (fmax - fmin) # type: ignore
        return states

    @property
    def prune_states(self):
        states = self._prune_states.copy()
        for i in range(states.shape[1]):
            fmin = self._pmin[i]
            fmax = self._pmax[i]
            if fmax - fmin > 0:
                states[:, i] = (states[:, i] - fmin) / (fmax - fmin)
        return states

    def _reset_everything(self):
        self._build_state()
        self.quant_strategy = Strategy(len(self.quant_idx), self.max_bit)
        self.prune_strategy = Strategy(
            len(self.prune_idx), self.max_heads)
        self.stage = Stage.Quant
        return self.reset()

    @property
    def strategy(self):
        return self.prune_strategy.strategy + self.quant_strategy.strategy

class DemoStepEnv:
    def __init__(self, l=20, lmin=3, rmax=15, target=[5.]*20):
        self.len = l
        self.list = [8.]*self.len
        self.cur_idx = 0
        self.min = lmin
        self.max = rmax
        self.target = target
        
    def step(self, action):
        action = self._action_wall(action)
        
        self.list[self.cur_idx] = action

        done = False
        reward = 0
        info = {'info': f"{self.cur_idx} take action {action}"}

        if self.is_final():
            done = True
            reward = self.reward()

        if not self.is_final():
            self.cur_idx += 1
        else:
            self.cur_idx =0

        state = np.array([self.cur_idx]+ self.list, dtype='float')
        return state, reward, done, info

    def reset(self):
        self.cur_idx = 0
        return np.array([self.cur_idx]+ self.list)

    def reward(self):
        target = np.array(self.target, dtype='float')
        return -sum(abs(np.array(self.list, dtype='float') - target)) / self.len

    def is_final(self):
        return self.cur_idx == self.len-1

    def _action_wall(self, action):
        action = float(action)
        lbound, rbound = self.min - 0.5, self.max + 0.5
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action

    @property
    def strategy(self):
        return deepcopy(self.list)


if __name__ == '__main__':
    path = r'D:\d-storage\output\vit\0.9853000044822693.pt'
    trainloader = get_cifar10_dataloader()
    testloader = get_cifar10_dataloader(train=False)
    env = Env(get_vit(args.QVIT), path, trainloader,
              testloader, 'cpu', args.ENV)
