from typing import Union
import torch
from torch.autograd.profiler_util import FunctionEvent, EventList
import numpy as np
from tqdm import tqdm
from gen import gen_dataset
import json
import matplotlib.pyplot as plt
import os
from model import *
from gen import *

GiGa = 1024 * 1024 * 1024


def self_flops(evt: FunctionEvent):
    flops = evt.flops
    for child in evt.cpu_children:
        child_flops = self_flops(child)
        if child_flops:
            flops += child_flops
    return flops


class EventAnalyser:
    def __init__(self, events: EventList) -> None:
        self.events = events

    @property
    def cpu_time_total(self):
        '''
        in ms \n
        1 s = 1e3 ms = 1e6 us
        '''
        return sum([evt.cpu_time_total for evt in self.events]) / 1e3

    @property
    def estimate_memory_usage(self):
        '''
        B
        '''
        max_mem = 0
        cur_mem = 0
        for evt in self.events:
            evt: FunctionEvent
            if evt.cpu_parent is None and evt.cpu_memory_usage != 0:
                if evt.name == '[memory]' and evt.cpu_memory_usage > 0:
                    continue
                cur_mem += evt.cpu_memory_usage
                max_mem = max(max_mem, cur_mem)
        return max_mem

    @property
    def max_memory_usage(self):
        '''
        KB = 1024B
        '''
        max_mem = 0
        cur_mem = 0
        for evt in self.events:
            evt: FunctionEvent
            if evt.cpu_parent is None and evt.cpu_memory_usage != 0:  # filter non-top-level op
                cur_mem += evt.cpu_memory_usage
                max_mem = max(max_mem, cur_mem)
        return max_mem

    def get_read_write_kflops_per_op(self):
        '''
        read/write: KByte
        flops: FLOPS
        '''
        self.events: list[FunctionEvent]
        op_dict = {}
        for evt in self.events:
            evt: FunctionEvent
            if evt.cpu_parent is None:  # filter non-top-level op
                op_dict[(evt.id, evt.name)] = {
                    'read': int(sum(np.prod(s) for s in evt.input_shapes if len(s) != 0)) * 4, # type: ignore
                    'write': evt.cpu_memory_usage,
                    'flops': self_flops(evt)
                }
        return op_dict

    @staticmethod
    def get_top_level_evts(events: EventList):
        evtlst = EventList()
        for evt in events:
            evt: FunctionEvent
            if evt.cpu_parent is None:  # filter non-top-level op
                evtlst.append(evt)
        return evtlst

    @property
    def total_flops(self):
        flops = 0
        for evt in self.events:
            if evt.cpu_parent is None:
                evt_flops = self_flops(evt)
                if evt_flops:
                    flops += evt_flops
        return flops

    @property
    def total_read(self):
        return int(sum(
            sum(np.prod(s) for s in evt.input_shapes if len(s) != 0) for evt in self.events if evt.cpu_parent is None)) * 4 // 1024 # type: ignore

    @property
    def total_write(self):
        return sum(evt.cpu_memory_usage for evt in self.events if evt.cpu_parent is None and evt.cpu_memory_usage > 0) // 1024

    def __repr__(self):
        s = "Name".ljust(20)+"CPU Time".ljust(15) + \
            "CPU mem".ljust(15)+"FLOPs".ljust(15)+"Input Shapes"
        for evt in self.events:
            evt: FunctionEvent
            if evt.cpu_parent is None:
                s += f"\n{evt.name}".ljust(20)+f"{evt.cpu_time_total_str}".ljust(
                    15)+f"{evt.cpu_memory_usage//1024}KB".ljust(15)+f'{self_flops(evt)}'.ljust(15)+f"{evt.input_shapes}"
        return s

    def print_meta(self):
        print(f"cpu time: {self.cpu_time_total}ms\nmax mem usage: {self.max_memory_usage}KB\ntotal mem read: {self.total_read}KB\ntotal mem write: {self.total_write}KB\ntotal flops: {self.total_flops}")


class ModuleProfiler(EventAnalyser):
    def __init__(self, module:nn.Module, x, num_threads=1, **kwargs) -> None:
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

        self.module = module
        self.x = x
        self.kwargs = kwargs
        super().__init__(self.__profile())

    def __profile(self):
        layer = self.module
        x = self.x

        with torch.no_grad():
            layer.eval()
            with torch.autograd.profiler.profile(with_flops=True, profile_memory=True, record_shapes=True) as prof:
                layer(x, **self.kwargs)
        events: EventList = prof.function_events  # type: ignore
        return events


class Profiler(ModuleProfiler):
    def __init__(self, layer_type, hparams, x_shape, num_threads=1, **kwargs) -> None:
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

        self.kwargs = kwargs
        self.layer_type = layer_type
        self.hparams = hparams
        self.x_shape = x_shape

        layer = self.layer_type(**self.hparams).eval()
        x = torch.randn(*self.x_shape)

        super().__init__(layer, x, num_threads, **kwargs)

    @property
    def energy_consumption(self):
        basic_coeff = {
            MlpBlock: 4.1,
            Linear: 4.,
            LinearGeneral: 3.95,
            SelfAttention: 3.4,
            EncoderBlock: 3.54
        }
        b = 2.5
        var = 0.
        coeff = 5.
        if self.layer_type in basic_coeff:
            coeff = basic_coeff[self.layer_type]  # type: ignore
        return self.cpu_time_total * (coeff-b-var*np.random.randn())


def prof(layer_type, data, num_threads=1):
    label = {}
    for idx in tqdm(data, desc="profile"):
        hparam = data[idx]['hparams']
        in_dim = data[idx]['x_shape']
        kwargs = {'dims': ([2], [0])} if layer_type is LinearGeneral else {}
        prof = Profiler(layer_type, hparam, in_dim,
                        num_threads=num_threads, **kwargs)
        label[idx] = {'time': prof.cpu_time_total, 'mem': prof.max_memory_usage, 'read': prof.total_read,
                      'write': prof.total_write, 'flops': prof.total_flops, 'emem': prof.estimate_memory_usage, 'e': prof.energy_consumption}
    return label


def train(x: np.ndarray, y: np.ndarray):
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    return coefficients


def pred(x: np.ndarray, coeff: np.ndarray):
    y_pred = x.dot(coeff)
    return y_pred


def mape(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.subtract(label, pred)) / label) * 100


def test(x, y, coeff: Union[np.ndarray, None]):
    '''
    return x, y, y_pred, mape(y, y_pred)
    '''
    x = np.array(x)
    y = np.array(y)

    if coeff is None:
        coeff = train(x, y)

    y_pred = pred(x, coeff)

    return x, y, coeff, y_pred, mape(y, y_pred)


def test_lat(label: dict, coeff=None):
    y, x = [], []
    for v in label.values():
        y.append(v['time'])
        x.append([v['read'], v['write'], v['flops']])
    return test(x, y, coeff)


def test_mem(label):
    mem, emem = [], []
    for v in label.values():
        mem.append(v['mem'])
        emem.append(v['emem'])
    return mape(np.array(mem), np.array(emem))


def test_e(label, coeff_lat, coeff_e=None):
    x, y = [], []
    for v in label.values():
        y.append(v['e'])
        x.append([v['read'], v['write'], v['flops']])
    x = pred(np.array(x), coeff_lat).reshape(len(x), 1)
    return test(x, y, coeff_e)


def plt_fig(y_fit, y_data, title, save_path, format='png', x_label='Measured', y_label='Predicted'):
    plt.figure()
    plt.title(title)
    plt.plot([min(y_data), max(y_data)], [
             min(y_data), max(y_data)], '--', color='gray')
    plt.scatter(y_data, y_fit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path, format=format)


def main(types, aliases, metrics=('lat', 'e'), train_only=False, n=200, data_save_dir=None, no_regenerate=True, need_plt=True, plt_save_dir=None):
    '''
    supported_metrics = ('lat', 'e')\n
    supported_types = (LinearGeneral, Linear, SelfAttention, MlpBlock, EncoderBlock)
    '''

    # assertions
    assert len(types) == len(aliases)

    supported_types = (LinearGeneral, Linear, SelfAttention,
                       MlpBlock, EncoderBlock)

    train_data = {alias: {} for alias in aliases}
    train_label = {alias: {} for alias in aliases}
    test_data = {alias: {} for alias in aliases}
    test_label = {alias: {} for alias in aliases}

    # 生成数据集并measure
    for type, alias in zip(types, aliases):
        if type not in supported_types:
            raise NotImplementedError()

        if data_save_dir:
            paths = {
                'train-data': os.path.join(data_save_dir, f'{alias}-train-data.json'),
                'train-label': os.path.join(data_save_dir, f'{alias}-train-label.json'),
                'test-data': os.path.join(data_save_dir, f'{alias}-test-data.json'),
                'test-label': os.path.join(data_save_dir, f'{alias}-test-label.json'),
            }

            if no_regenerate:
                train_data[alias] = json.load(
                    open(paths['train-data'], 'r')) if os.path.exists(paths['train-data']) else {}
                train_label[alias] = json.load(
                    open(paths['train-label'], 'r')) if os.path.exists(paths['train-label']) else {}

                if not train_only:
                    test_data[alias] = json.load(
                        open(paths['test-data'], 'r')) if os.path.exists(paths['test-data']) else {}
                    test_label[alias] = json.load(
                        open(paths['test-label'], 'r')) if os.path.exists(paths['test-label']) else {}

            if len(train_data[alias]) == 0:
                train_data[alias] = gen_dataset(type, size=50)
                train_label[alias] = prof(type, train_data[alias])

                if not os.path.exists(data_save_dir):
                    os.makedirs(data_save_dir)
                json.dump(train_data[alias], open(
                    paths['train-data'], 'w'), indent=2)
                json.dump(train_label[alias], open(
                    paths['train-label'], 'w'), indent=2)

            if len(test_data[alias]) == 0:
                test_data[alias] = gen_dataset(type, size=n)
                test_label[alias] = prof(type, test_data[alias])

                if not os.path.exists(data_save_dir):
                    os.makedirs(data_save_dir)
                json.dump(test_data[alias], open(
                    paths['test-data'], 'w'), indent=2)
                json.dump(test_label[alias], open(
                    paths['test-label'], 'w'), indent=2)

        if len(train_data[alias]) == 0:
            train_data[alias] = gen_dataset(type, size=50)
            train_label[alias] = prof(type, train_data[alias])
        if len(test_data[alias]) == 0 and not train_only:
            test_data[alias] = gen_dataset(type, size=n)
            test_label[alias] = prof(type, test_data[alias])

    coeff_lat = {}
    coeff_e = {}

    result = {alias: {} for alias in aliases}

    def eval_lat(alias, cl):
        r = {}
        x, y, cl, y_pred, train_mape = test_lat(train_label[alias], cl)
        r['coeff_lat'] = cl.tolist()
        r['train_lat_mape'] = train_mape

        if need_plt:
            assert plt_save_dir is not None
            plt_fig(
                y_pred, y,
                title=f'Train Set {alias} Latency(ms)',
                save_path=os.path.join(plt_save_dir, f'{alias}-train-lat.png')
            )

        if not train_only:
            x, y, cl, y_pred, test_mape = test_lat(test_label[alias], cl)
            r['test_lat_mape'] = test_mape
            if need_plt:
                assert plt_save_dir is not None
                plt_fig(
                    y_pred, y,
                    title=f'Test Set {alias} Latency(ms)',
                    save_path=os.path.join(
                        plt_save_dir, f'{alias}-test-lat.png')
                )
        return r, cl

    def eval_e(alias, cl, ce):
        r = {}
        x, y, ce, y_pred, train_mape = test_e(train_label[alias], cl, ce)
        r['coeff_e'] = ce.tolist()
        r['train_e_mape'] = train_mape

        if need_plt:
            assert plt_save_dir is not None
            plt_fig(
                y_pred, y,
                title=f'Train Set {alias} Energy Consumption(mJ)',
                save_path=os.path.join(plt_save_dir, f'{alias}-train-e.png')
            )
        if not train_only:
            x, y, cl, y_pred, test_mape = test_e(test_label[alias], cl, ce)
            r['test_e_mape'] = test_mape
            if need_plt:
                assert plt_save_dir is not None
                plt_fig(
                    y_pred, y,
                    title=f'Test Set {alias} Energy Consumption(J)',
                    save_path=os.path.join(plt_save_dir, f'{alias}-test-e.png')
                )
        return r, cl

    if 'lat' in metrics:
        for alias in aliases:
            info, coeff_lat[alias] = eval_lat(
                alias, coeff_lat[alias] if alias in coeff_lat else None)
            result[alias].update(info)

    if 'e' in metrics:
        for alias in aliases:
            if alias not in coeff_lat:
                _, coeff_lat[alias] = eval_lat(alias, None)
            info, coeff_e[alias] = eval_e(
                alias, coeff_lat[alias], coeff_e[alias] if alias in coeff_e else None)
            result[alias].update(info)
    return result


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


if __name__ == '__main__':
    # types = (LinearGeneral, Linear, SelfAttention, MlpBlock, EncoderBlock)
    # aliases = ('LinearGeneral', 'Linear', 'MSA', 'FFN', 'Encoder')
    # result = main(types, aliases, metrics=('lat', 'e'),
    #               data_save_dir='tmp', plt_save_dir='plt', no_regenerate=False)

    # json.dump(result, open('result.json', 'w'), indent=2)
    # in_dim = 768
    # mlp_dim = 3078
    # head_dim = 64
    # x = torch.randn(1, 197, 768)
    # encoder1 = EncoderBlock(in_dim, mlp_dim, num_heads=6, head_dim=head_dim)
    # encoder2 = EncoderBlock(in_dim, mlp_dim, num_heads=12, head_dim=head_dim)
    # mp1 = ModuleProfiler(encoder1, x)
    # mp2 = ModuleProfiler(encoder2, x)
    # print(mp2.cpu_time_total)
    coeff_lat = [
      -0.0007415096412637279,
      0.005590396846728316,
      3.352617495357331e-08
    ]
    coeff_e = 1.0399999999770126
    rs = min_max_policy_consum(coeff_lat, coeff_e, 12, 3, 12, 4, 8, 32)

    for r in rs:
        lat, e, mem, w = r
        print(f'lat: {lat/1000}s, e: {e/1000}J, mem: {mem/1000}MB, weight size: {w/1000}MB')