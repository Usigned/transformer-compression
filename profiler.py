from model import EncoderBlock, CAFIA_Transformer, SelfAttention, LinearGeneral, EncoderBlock
from torch.profiler import profile, record_function, ProfilerActivity
from argparse import Namespace
import json
import torch
from torch.autograd.profiler_util import FunctionEvent

torch.set_num_threads(1)

pwd = '/Users/qing/Documents/design-code/ViT-B_16-224.json'
args = Namespace(**json.load(open(pwd, 'r')))
args.attn_type = SelfAttention

# model = CAFIA_Transformer(args)
# inputs = torch.randn(1, 3, 224, 224)

# model = LinearGeneral((162,), (3, 54))
# inputs = torch.randn(1, 192, 162)
# print(1*192*162+162*3*54+192*3*54)
model = SelfAttention(768, heads=12)
# model = EncoderBlock(768, 3072, 12).eval()
inputs = torch.randn(1, 197, 768)

# with profile(activities=[ProfilerActivity.CPU], with_flops=True, profile_memory=True) as prof:
#     model(inputs)

# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))

# inputs = torch.randn(1, 192, 162)

with torch.autograd.profiler.profile(profile_memory=True, with_flops=True) as prof:
    model(inputs)

for evt in prof.function_events:
    evt:FunctionEvent
    if evt.cpu_parent is None and evt.cpu_memory_usage != 0:
        print(
        f'''{evt.id}, {evt.name}, {evt.cpu_memory_usage}, {evt.cpu_time_total_str}, {evt.input_shapes}, {evt.self_flops}
        '''
        )

# num_threads = torch.get_num_threads()
# print(f'Benchmarking on {num_threads} threads')

# import torch.utils.benchmark as benchmark

# t = benchmark.Timer(
#     stmt='model(inputs)',
#     globals={'inputs': inputs, 'model': model}
# )

# print(t.timeit(100).mean*1000)
