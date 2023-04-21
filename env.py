from collections import namedtuple
import math
from typing import Union
import torch.optim as optim
import torch
import torch.nn as nn
from train import finetune, eval_model
from quant_utils import *
from model import *
from enum import Enum
import numpy as np
import args
from copy import deepcopy
from mmsa import *

State = namedtuple('State', 'method, idx, num_heads, in_dim, out_dim, prec')


# copy from profiler, pytorch 1.81 doesn't have some neccerary imports
def min_max_policy_consum(coeff_lat, coeff_e, num_encoders, min_heads, max_heads, min_b, max_b, a_b):
    assert len(coeff_lat) == 3
    _r, _w, _f = coeff_lat

    (m_r, m_w, m_f), m_ir, m_w = encoder_summary(197, 768, min_heads, 3078, 64, a_b, w_b=[min_b]*4)

    min_lat = (m_r * _r + m_w * _w + m_f * _f) * num_encoders
    min_e = coeff_e * min_lat *num_encoders
    min_w = m_w * num_encoders
    min_mem = m_w * num_encoders + m_ir
    (m_r, m_w, m_f), m_ir, m_w = encoder_summary(197, 768, max_heads, 3078, 64, a_b, w_b=[max_b]*4)

    max_lat = (m_r * _r + m_w * _w + m_f * _f) * num_encoders
    max_e = coeff_e * min_lat *num_encoders
    max_w = m_w * num_encoders
    max_mem = m_w * num_encoders + m_ir

    return (min_lat, min_e, min_mem, min_w), (max_lat, max_e, max_mem, max_w)


def encoder_summary(seq_len, in_dim, heads, mlp_dim, head_dim, a_b=32, w_b:Union[int, list]=32):
    '''
    (r, w, flops), mem, w_size\n
    return in KB = 1000 Byte = 1024 * 8 bit\n
    fp32 = 4Byte, 1 Byte = 8 bit
    '''
    matrix_size = seq_len**2 * heads
    in_size = seq_len * in_dim
    seq_heads_size = seq_len * heads * head_dim
    mlp_w_size = mlp_dim * in_dim
    lg_w_size = in_dim * heads * head_dim
    msa_max_mem = seq_heads_size * 4 + matrix_size + in_size*2
    ffn_max_mem = in_size + mlp_dim * seq_len * 2
    max_mem = (max(ffn_max_mem, msa_max_mem) + in_size) * a_b // 8 //1024

    def encoder_weight():
        wbs = []
        if isinstance(w_b, int):
            wbs = [w_b] * 4
        else:
            wbs = w_b
        assert len(wbs) == 4
        weight_size = 0
        weight_size += lg_w_size * 3 * wbs[0]
        weight_size += lg_w_size * wbs[1]
        weight_size += mlp_w_size * wbs[2]
        weight_size += mlp_w_size * wbs[3]
        return weight_size //8 // 1024

    def encoder_runtime():
        read = write = flops = 0
        # layer norm
        read += in_dim * seq_len + in_dim * 2
        write += in_dim * seq_len
        flops += 0

        # msa in: linear general * 3
        for _ in range(3):
            #            x                        w                         x                  b
            read += in_dim * seq_len + in_dim * heads * head_dim
            #              matmul             add
            write += seq_heads_size
            #                       matmul                           add
            flops += in_dim * seq_len * heads * head_dim * 2

            read  += seq_heads_size +  heads * head_dim
            write +=  heads * head_dim * seq_len
            flops +=   heads * head_dim * seq_len



        # reshape, transpose
        read += seq_len * heads * head_dim * 4

        # q * k
        read += seq_len * heads * head_dim * 2
        flops += 2 * seq_len * seq_len * heads * head_dim
        write += seq_len * seq_len * heads


        #scale, softmax
        read += seq_len * seq_len * heads
        write += seq_len * seq_len * heads

        read += seq_len ** 2 * heads
        write += seq_len ** 2 * heads

        # s * v
        read += seq_len**2 * heads + heads*seq_len*head_dim
        flops += 2 * seq_len * seq_len * heads * head_dim
        write += seq_len * heads * head_dim
        
        read += heads*head_dim*seq_len
        
        # msa out: linear general
        read += heads * head_dim * in_dim + heads *seq_len*head_dim + in_dim * seq_len + in_dim
        write += in_size + in_size
        flops += in_dim * seq_len * heads * head_dim * 2 + in_dim * seq_len

        #####################################
        # add_, layernorm
        read += in_size * 2 + in_size + in_dim * 2
        write += in_size
        # mlp fc1
        read += in_size + mlp_w_size + mlp_dim
        write += seq_len * mlp_dim
        flops += mlp_w_size * seq_len * 2
        #gelu
        read += mlp_dim * seq_len
        write += seq_len * mlp_dim
        read += mlp_dim * seq_len
        # mlp fc2
        read += mlp_w_size + mlp_dim*seq_len + in_dim
        flops += mlp_w_size * seq_len * 2
        write += in_size
        # add_
        read += in_size * 2

        return read * a_b // 8 //1024, write * a_b // 8 //1024, flops

    return encoder_runtime(), max_mem, encoder_weight()

def estimate_encoder_lat_e(coeff_lat, coeff_e, r, w, flops):
    assert len(coeff_lat) == 3
    lat = r * coeff_lat[0] + w*coeff_lat[1] +flops*coeff_lat[2]
    e = lat * coeff_e
    return lat, e 



class Stage(Enum):
    Quant = 0
    Prune = 1


class QuantPruneEnv:
    def __init__(self, model: CAFIA_Transformer, weight_path, trainloader, testloader, lat_b, e_b, mem_b, min_bit, max_bit, a_bit, max_heads, min_heads, head_dim, ori_acc, device, state_dim=7, float_bit=8, prune_only=False) -> None:
        
        self.prune_only = prune_only
        self.model = model
        self.weight_path = weight_path
        self.load_weight()
        self._model = deepcopy(model)

        self.trainloader = trainloader
        self.testloader = testloader

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

        # resouce bound
        self.lat_b = lat_b
        self.e_b = e_b
        self.mem_b = mem_b
        self.__init_resource_computation()

    def __init_resource_computation(self):
        self.coeff_lat = args.COEF_LAT
        self.coeff_e = args.COEF_E
        _min_resource = QuantPruneEnv.estimate_strategy(
            [self.min_bit]*len(self.quant_strategy),
            [self.min_heads]*len(self.prune_strategy),
            self.coeff_lat,
            self.coeff_e,
            self.a_bit
        )
        self._possible_min_lat, self._possible_min_e, self._possible_min_mem= _min_resource
        print(_min_resource)
        self.__resource_assertion()

    def __resource_assertion(self):
        assert self._possible_min_lat < self.lat_b, f"{self._possible_min_lat} > {self.lat_b}"
        assert self._possible_min_e < self.e_b, f"{self._possible_min_e} > {self.e_b}"
        assert self._possible_min_mem < self.mem_b, f"{self._possible_min_mem} > {self.mem_b}"

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
        if not self.prune_only:
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

            s = f'{self.stage.name} finish\nprune_pi:{self.prune_strategy}\nreward: {reward}\n'
            if not self.prune_only: s += f'quant_pi:{self.quant_strategy}\n'
            info_set['info'] = s
            self._build_state()
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
        with open('strategy.his', 'a+') as f:
            acc = eval_model(self.model, self.testloader, self.device)
            f.write(f'{self.strategy} has acc {acc}\n')
            f.flush()

        return (acc - self.ori_acc) * 0.1

    def _adjust_strategy(self):
        self.__resource_assertion()
        prune_len = len(self.prune_strategy)
        quant_len = len(self.quant_strategy)
        idx= prune_len + quant_len - 1
        while not self._resource_bound_statisfied():
            if idx > prune_len-1:
                i = idx-prune_len
                if self.quant_strategy[i] > self.min_bit:
                    self.quant_strategy[i] -= 1
                if self._resource_bound_statisfied(): return
            else:
                j = idx
                if self.prune_strategy[j] > self.min_heads:
                    self.prune_strategy[j] -= 1
                if self._resource_bound_statisfied(): return
            idx -= 1
            if idx < 0: 
                idx = len(self.quant_strategy)-1

    def _estimate_strategy(self):
        return QuantPruneEnv.estimate_strategy(self.quant_strategy, self.prune_strategy, self.coeff_lat, self.coeff_e, self.a_bit)

    @staticmethod
    def estimate_strategy(q_s, pr_s, coeff_lat, coeff_e, a_bit=8):
        lat, e, mem = 0, 0, 0
        i, j = 0, 0
        _mir = 0
        while i < len(pr_s) and j < len(q_s):
            heads = pr_s[i]
            wbs = q_s[j:4+j]
            i += 1
            j += 4
            (r, w, flops), _mem, w_size = encoder_summary(197, 768, heads, 3078, 64, a_bit, wbs)
            _lat, _e = estimate_encoder_lat_e(coeff_lat, coeff_e, r, w, flops)
            lat += _lat
            e += _e
            mem += w_size
            _mir = max(_mir, _mem)
        _mir +=  197*768*4//1024
        mem += _mir
        return lat, e, mem

    def _resource_bound_statisfied(self):
        lat, e, mem = self._estimate_strategy()
        return lat < self.lat_b and e < self.e_b and self.mem_b > mem

    def reset(self):
        self.load_weight()
        self.cur_idx = 0
        self.stage = Stage.Quant if self.stage is Stage.Prune and not self.prune_only else Stage.Prune
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
        l = self.prune_strategy.tolist()
        if not self.prune_only:
            l += self.quant_strategy.tolist()
        return l


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
        return -sum(abs(np.array(self.list, dtype='float') - target)) / self.len  # type: ignore

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
