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
from mmsa import *

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
        assert len(
            self) == self.target_len, f"len: {len(self)}, target: {self.target_len}, val: {self.cur_strategy}"
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
    def __init__(self, model: CAFIA_Transformer, weight_path, trainloader, testloader, lat_b, e_b, mem_b, min_bit, max_bit, a_bit, max_heads, min_heads, head_dim, ori_acc, device, state_dim=7, float_bit=8) -> None:
        self.model = model
        self.weight_path = weight_path
        self.load_weight()
        self._model = deepcopy(model)

        self.trainloader = trainloader
        self.testloader = testloader

        # resouce bound
        self.lat_b = lat_b
        self.e_b = e_b
        self.mem_b = mem_b

        # quant bit range
        self.min_bit = min_bit
        self.max_bit = max_bit
        self.a_bit = a_bit

        # prune heads
        self.max_heads = max_heads
        self.min_heads = min_heads
        self.head_dim = head_dim

        self.device = device
        self.ori_acc = ori_acc
        self.float_bit = float_bit
        self.state_dim = state_dim

        self.init_env()
        # runtime
        self.cur_idx = 0
        self.best_reward = -math.inf
        self.stage = Stage.Quant


    def init_env(self):
        self.quant_idxs, quant_idx = [], []
        for idx, m in enumerate(self.model.modules()):
            if type(m) in (QLinearGeneral, QLinear):
                quant_idx.append(idx)
        i = 0
        while i < len(quant_idx):
            q, k, v, out, fc1, fc2 = quant_idx[i: i+6]
            self.quant_idxs += [[q, k, v], [out], [fc1], [fc2]]
            i += 6
        self.prune_idxs = [[idx] for idx, m in enumerate(
            self.model.modules()) if type(m) is LearnableMask]

        self.quant_strategy = self._new_strategy(quant=True)
        self.prune_strategy = self._new_strategy(quant=False)

        self._quant_states = self._new_states(len(self.quant_strategy))
        self._prune_states = self._new_states(len(self.prune_strategy))

        self._build_state()
        self._init_scales()

    def _new_states(self, num_states):
        return np.ndarray((num_states, self.state_dim), dtype='int')

    def _new_strategy(self, quant=True):
        data = len(self.quant_idxs) * \
            [self.float_bit] if quant else len(
                self.prune_idxs)*[self.max_heads]
        return np.array(data, dtype='int')

    def _build_state(self):
        msa_states = []
        fc_states = []
        for m in self.model.modules():
            if type(m) is MaskedSelfAttention:
                heads = m.heads - m.mask.pruned_dim
                q, k, v, out = m.query, m.key, m.value, m.out
                assert isinstance(q, QModule) and isinstance(
                    k, QModule) and isinstance(v, QModule) and isinstance(out, QModule)
                assert q.w_bit == k.w_bit == v.w_bit
                prec1 = q.w_bit if q.w_bit > 0 else self.float_bit
                prec2 = out.w_bit if out.w_bit > 0 else self.float_bit
                msa_state = [heads, m.head_dim*m.heads,
                             m.heads*m.head_dim, prec1, prec2]
                lg_qkv_state = [1, m.head_dim*m.heads,
                                heads*m.head_dim, prec1, prec1]
                lg_out_state = [1, m.head_dim*heads,
                                m.heads*m.head_dim, prec1, prec1]
                msa_states.append(msa_state)
                fc_states.append(lg_qkv_state)
                fc_states.append(lg_out_state)

            if type(m) is QLinear:
                heads = 1
                in_dim = m.in_features
                out_dim = m.out_features
                b = m.w_bit if m.w_bit > 0 else self.float_bit
                fc_states.append([heads, in_dim, out_dim, b, b])
        for idx, state in enumerate(msa_states):
            self._prune_states[idx] = np.array([0, idx]+state, dtype='int')

        for idx, state in enumerate(fc_states):
            self._quant_states[idx] = np.array([1, idx]+state, dtype='int')

    def _apply_strategy(self):
        self._apply_prune()
        self._apply_quant()

    @property
    def _prune_idx_strategy(self):
        prune_idx = []
        for idxs in self.prune_idxs:
            for idx in idxs:
                prune_idx.append(idx)
        return prune_idx, self.prune_strategy

    def _apply_prune(self):
        mix_prune(self.model, *self._prune_idx_strategy)

    @property
    def _quant_idx_strategy(self):
        quant_idx = []
        quant_strategy = []
        for idxs, b in zip(self.quant_idxs, self.quant_strategy):
            for idx in idxs:
                quant_strategy.append(b)
                quant_idx.append(idx)
        return quant_idx, mix_weight_act_strategy(
            quant_strategy, [self.a_bit]*len(quant_strategy))

    def _apply_quant(self):
        assert len(self.quant_idxs) == len(self.quant_strategy)
        set_mixed_precision(self.model, *self._quant_idx_strategy)

    def step(self, action):
        action = self._action_wall(action)

        if self.stage is Stage.Quant:
            self.quant_strategy[self.cur_idx] = action
        else:
            self.prune_strategy[self.cur_idx] = action

        info_set = {'info': f"{self.stage.name} take action {action}"}

        if self._is_batch_end():
            self._adjust_strategy()

            self._apply_strategy()
            reward = self.reward()
            done = True

            info_set['info'] = f'{self.stage.name} finish\nquant_pi:{self.quant_strategy}\nprune_pi:{self.prune_strategy}\nreward: {reward}\n'


            next_state = self.reset()
            return next_state, reward, done, info_set

        self.cur_idx += 1
        next_state = self._get_next_states()
        reward = 0
        done = False

        return next_state, reward, done, info_set


    def _get_next_states(self, norm=False):
        if self.stage is Stage.Quant:
            return self.quant_states[self.cur_idx, :].copy() if norm else self._quant_states[self.cur_idx, :].copy()
        return self.prune_states[self.cur_idx, :].copy() if norm else self._prune_states[self.cur_idx, :].copy()

    def reward(self):
        if True:
            finetune(self.model, self.trainloader, self.device)
        acc = eval_model(self.model, self.testloader, self.device)

        return (acc - self.ori_acc) * 0.1

    def _adjust_strategy(self):
        pass

    def reset(self):
        self.load_weight()
        self.cur_idx = 0
        self.stage = Stage.Quant if self.stage is Stage.Prune else Stage.Prune
        self._build_state()
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
            return self.cur_idx == len(self.quant_strategy)-1
        if self.stage is Stage.Prune:
            return self.cur_idx == len(self.prune_strategy)-1

    @property
    def quant_states(self):
        states = self._quant_states.copy()
        for i in range(states.shape[1]):
            fmin = self._qmin[i]
            fmax = self._qmax[i]
            if fmax - fmin > 0:
                states[:, i] = (states[:, i] - fmin) / \
                    (fmax - fmin)  # type: ignore
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

    def _init_scales(self):
        '''
        should compute only once during init
        '''
        _qmin = [1, 0, 1]
        _qmin.append(
            min(min(self._quant_states[:, 3]), self.min_heads*self.head_dim))
        _qmin.append(
            min(min(self._quant_states[:, 4]), self.min_heads*self.head_dim))
        _qmin.append(self.min_bit)
        _qmin.append(self.min_bit)
        self._qmin = _qmin

        _qmax = [1, max(self._quant_states[:, 1]), 1]
        _qmax.append(
            max(max(self._quant_states[:, 3]), self.max_heads*self.head_dim))
        _qmax.append(
            max(max(self._quant_states[:, 4]), self.max_heads*self.head_dim))
        _qmax.append(self.max_bit)
        _qmax.append(self.max_bit)
        self._qmax = _qmax

        # prune state scale
        _pmin = [1, 0, self.min_heads]
        _pmin.append(
            min(min(self._prune_states[:, 3]), self.min_heads*self.head_dim))
        _pmin.append(
            min(min(self._prune_states[:, 4]), self.min_heads*self.head_dim))
        _pmin.append(self.min_bit)
        _pmin.append(self.min_bit)
        self._pmin = _pmin

        _pmax = [1, max(self._prune_states[:, 1]), self.max_heads]
        _pmax.append(
            max(max(self._prune_states[:, 3]), self.max_heads*self.head_dim))
        _pmax.append(
            max(max(self._prune_states[:, 4]), self.max_heads*self.head_dim))
        _pmax.append(self.max_bit)
        _pmax.append(self.max_bit)
        self._pmax = _pmax

    def _reset_everything(self):
        self._build_state()
        self.quant_strategy = self._new_strategy(quant=True)
        self.prune_strategy = self._new_strategy(quant=False)
        self.stage = Stage.Quant
        return self.reset()

    @property
    def strategy(self):
        return self.prune_strategy.tolist() + self.quant_strategy.tolist()


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
            self.cur_idx = 0

        state = np.array([self.cur_idx] + self.list, dtype='float')
        return state, reward, done, info

    def reset(self):
        self.cur_idx = 0
        return np.array([self.cur_idx] + self.list)

    def reward(self):
        target = np.array(self.target, dtype='float')
        # type: ignore
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
