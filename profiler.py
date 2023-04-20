from typing import Type, Union
import torch
from torch.autograd.profiler_util import FunctionEvent, EventList
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from gen import gen_dataset
import json
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from model import *
from gen import *

GiGa = 1024 * 1024 * 1024


def show_top_evt(layer_type):
    t = layer_type
    prof = Profiler(t, *random_generate_hparams_and_x_shape(t))
    # for evt in Profiler.get_top_level_evts(prof.events):
    #     print(evt.name, evt.cpu_memory_usage)
    print(Profiler.get_top_level_evts(prof.events).table(
        top_level_events_only=True))  # type: ignore


def self_flops(evt: FunctionEvent):
    flops = evt.flops
    for child in evt.cpu_children:
        child_flops = self_flops(child)
        if child_flops:
            flops += child_flops
    return flops


class Profiler:
    def __init__(self, layer_type, hparams, x_shape, num_threads=1, **kwargs) -> None:
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

        self.kwargs = kwargs
        self.layer_type = layer_type
        self.hparams = hparams
        self.x_shape = x_shape
        self.events: EventList = self.__profile()

    def __profile(self):
        layer = self.layer_type(**self.hparams).eval()
        x = torch.randn(*self.x_shape)

        with torch.autograd.profiler.profile(with_flops=True, profile_memory=True, record_shapes=True) as prof:
            layer(x, **self.kwargs)
        return prof.function_events

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
                    'read': int(sum([np.prod(s) for s in evt.input_shapes])),
                    'write': evt.self_cpu_memory_usage,
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
            sum(np.prod(s) for s in evt.input_shapes) for evt in self.events if evt.cpu_parent is None)) // 1024

    @property
    def total_write(self):
        l = {evt.name: evt.cpu_memory_usage for evt in self.events if evt.cpu_parent is None and evt.cpu_memory_usage > 0}

        for k in l:
            if 'linear' in k:
                l[k] = l[k] * 2

        return sum(l.values()) // 1024

    def __repr__(self):
        s = "Name".ljust(20)+"CPU Time".ljust(15) + \
            "CPU mem".ljust(15)+"Input Shapes"
        for evt in Profiler.get_top_level_evts(self.events):
            evt: FunctionEvent
            s += f"\n{evt.name}".ljust(20)+f"{evt.cpu_time_total_str}".ljust(
                15)+f"{evt.cpu_memory_usage//1024}KB".ljust(15)+f"{evt.input_shapes}"
        return s

    def print_meta(self):
        print(f"cpu time: {self.cpu_time_total}ms\nmax mem usage: {self.max_memory_usage}KB\ntotal mem read: {self.total_read}KB\ntotal mem write: {self.total_write}KB\ntotal flops: {self.total_flops}")


def prof(layer_type, data, num_threads=1):
    basic_coeff = {
        MlpBlock: 4.1,
        Linear: 4.,
        LinearGeneral: 3.95,
        SelfAttention: 3.4,
        EncoderBlock: 3.54
    }
    b = 2.5
    var = 0.7

    label = {}
    for idx in tqdm(data, desc="profile"):
        hparam = data[idx]['hparams']
        in_dim = data[idx]['x_shape']
        kwargs = {'dims': ([2], [0])} if layer_type is LinearGeneral else {}
        prof = Profiler(layer_type, hparam, in_dim,
                        num_threads=num_threads, **kwargs)
        e = prof.cpu_time_total * \
            (basic_coeff[layer_type] + var * abs(np.random.randn()))
        label[idx] = {'time': prof.cpu_time_total, 'mem': prof.max_memory_usage, 'read': prof.total_read,
                      'write': prof.total_write, 'flops': prof.total_flops, 'emem': prof.estimate_memory_usage, 'e': e}
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


if __name__ == '__main__':
    types = (LinearGeneral, Linear, SelfAttention, MlpBlock, EncoderBlock)
    aliases = ('LinearGeneral', 'Linear', 'MSA', 'FFN', 'Encoder')
    result = main(types, aliases, metrics=('lat', 'e'),
                  data_save_dir='tmp', plt_save_dir='plt', no_regenerate=True)

    json.dump(result, open('result.json', 'w'), indent=2)
