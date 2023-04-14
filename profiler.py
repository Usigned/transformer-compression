from typing import Type
import torch
from torch.autograd.profiler_util import FunctionEvent, EventList
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from gen import gen_dataset
import json
import matplotlib.pyplot as plt

GiGa = 1024 * 1024 * 1024

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
        x  = torch.randn(*self.x_shape)

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


def profile(layer_type, data, num_threads=1):
    label = {}
    for idx in tqdm(data, desc="profile"):
        hparam = data[idx]['hparams']
        in_dim = data[idx]['x_shape']
        kwargs = {'dims': ([2], [0])} if layer_type is LinearGeneral else {}
        prof = Profiler(layer_type, hparam, in_dim, num_threads=num_threads, **kwargs)
        label[idx] = {'time': prof.cpu_time_total, 'mem': prof.max_memory_usage, 'read': prof.total_read, 'write':prof.total_write, 'flops':prof.total_flops}
    return label

def fit_and_plt(label, fname, need_plt=True):

    y, x = [], []
    for v in label.values():
        y.append(v['time'])
        x.append([v['read'], v['write'], v['flops']])

    # 假设我们有以下的数据点
    X = np.array(x)
    y_data = np.array(y)

    # 使用线性代数方法进行多元线性拟合
    # X = np.hstack((X, np.ones((X.shape[0], 1))))  # 添加常数列到 x 数据中
    coefficients = np.linalg.lstsq(X, y_data, rcond=None)[0]  # 最小二乘法拟合

    y_fit = X.dot(coefficients)

    if need_plt:
        plt.figure()
        plt.title(f"{fname} train")
        plt.plot([min(y_data), max(y_data)], [min(y_data), max(y_data)], '--', color='gray')
        plt.scatter(y_data, y_fit)
        plt.xlabel('True Label')
        plt.ylabel('Prediction')
        plt.savefig(f"{fname}-train.png", format='png')

    # if need_plt:
    #     indices = np.argsort(y_data)
    #     X_sorted = X[indices]
    #     y_sorted = y_data[indices]

    #     # 计算拟合曲线的值
    #     # X_sorted = np.hstack((X_sorted, np.ones((X_sorted.shape[0], 1))))  # 添加常数列到排序后的 x 数据中
    #     y_fit = X_sorted.dot(coefficients)

    #     # 绘制数据点和拟合曲线
    #     plt.figure()
    #     plt.title(f"{fname} train")
    #     plt.scatter(range(len(y_sorted)), y_sorted, label="data", color='black')
    #     plt.plot(range(len(y_fit)), y_fit, label="fit", color='black')
    #     plt.legend()
    #     plt.savefig(f"{fname}-train.png", format='png')

    return coefficients

def pred_and_plt(label, coefficients, fname, need_plt=True):
    y, x = [], []
    for v in label.values():
        y.append(v['time'])
        x.append([v['read'], v['write'], v['flops']])

    # 假设我们有以下的数据点
    X = np.array(x)
    y_data = np.array(y)

    # 使用线性代数方法进行多元线性拟合
    # X = np.hstack((X, np.ones((X.shape[0], 1))))  # 添加常数列到 x 数据中
    coefficients = np.linalg.lstsq(X, y_data, rcond=None)[0]  # 最小二乘法拟合

    y_fit = X.dot(coefficients)
    
    mer = np.mean(np.abs(y_data - y_fit) / y_data) * 100
    if need_plt:
        plt.figure()
        plt.title(f"{fname} test")
        plt.plot([min(y_data), max(y_data)], [min(y_data), max(y_data)], '--', color='gray')
        plt.scatter(y_data, y_fit)
        plt.xlabel('True Label')
        plt.ylabel('Prediction')
        plt.savefig(f"{fname}-test.png", format='png')

    # if need_plt:
    #     # 绘制数据点和拟合曲线
    #     plt.figure()
    #     plt.title(f"{fname} test")
    #     plt.scatter(range(len(y_sorted)), y_sorted, label="Data", color='black')
    #     plt.plot(range(len(y_fit)), y_fit, label="Fit", color='black')
    #     plt.legend()
    #     plt.savefig(f"{fname}-pred.png", format='png')
    return mer

def gen_and_profile_and_pred_and_plt(layer_type, n=200, need_plt=True):
    data = gen_dataset(layer_type, size=n)
    label = profile(layer_type, data)

    fname = str(layer_type).split('\'')[1].split('.')[-1]

    json.dump(data, open(f"{fname}-data.json", 'w'), indent=4)
    json.dump(label, open(f"{fname}-label.json", 'w'), indent=4)
    coeff = fit_and_plt(label, fname, need_plt=need_plt)
    return coeff

def test(layer_type, coeff, n=200):
    data = gen_dataset(layer_type, size=n)
    label = profile(layer_type, data)

    fname = str(layer_type).split('\'')[1].split('.')[-1]
    return pred_and_plt(label, coeff, fname)


if __name__ == '__main__':
    from model import LinearGeneral, SelfAttention, EncoderBlock, MlpBlock, Linear
    from gen import random_generate_hparams_and_x_shape
    types = [LinearGeneral, SelfAttention, EncoderBlock, MlpBlock, Linear]

    # f = open('coeff.csv', 'w')

    # for t in types:
    #     fname = str(t).split('\'')[1].split('.')[-1]
    #     coeff= gen_and_profile_and_pred_and_plt(t, n=50, need_plt=True)
    #     f.write(f'{fname},{coeff}\n')
    #     test(t, coeff, 200)

    # f.close()
    
    # t = EncoderBlock
    t = SelfAttention
    prof = Profiler(t, *random_generate_hparams_and_x_shape(t))

    # for evt in Profiler.get_top_level_evts(prof.events):
    #     print(evt.name, evt.cpu_memory_usage)
        # print(evt)
    print(prof.estimate_memory_usage)
    print(prof.max_memory_usage)
