import torch
import torch.nn as nn
from model import LinearGeneral, MlpBlock, SelfAttention, EncoderBlock

import time

def timeit1(fn, *args, repeat=1000, **kwargs):
    t = 0
    for i in range(repeat):
        t1 = time.thread_time()
        fn(*args, **kwargs)
        t2 = time.thread_time()
        t = t + t2 - t1
    return t

def timeit2(fn, *args, repeat=1000, **kwargs):
    t1 = time.thread_time()
    for i in range(repeat):
        fn(*args, **kwargs)
    t2 = time.thread_time()
    return (t2 - t1) * 1000 / repeat

if __name__ == "__main__":
    x = torch.randn(1, 197, 768)
    y = torch.randn(1, 768, 197)
    print("time1: ", timeit1(torch.matmul, x, y))
    print("time2:", timeit2(torch.matmul, x, y))