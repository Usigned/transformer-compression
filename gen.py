import torch
import torch.nn as nn
from model import LinearGeneral, MlpBlock, SelfAttention, EncoderBlock
from timeIt import timeit
import random
import json
from tqdm import tqdm
from utils import size

hyper_param_range = {
    nn.Linear: {
        "in_features": (200, 3500),
        "out_features": (200, 3500),
    },
    SelfAttention: {
        "heads": (1, 15),
        "attn_dim": (50, 75)
    },
    MlpBlock: {
        "in_dim": (500, 1000),
        "mlp_dim": (2750, 3150),
        "out_dim": (500, 1000)
    },
    EncoderBlock: {
        "attn_dim": (50, 75),
        "mlp_dim": (2750, 3150),
        "num_heads": (1, 15)
    },
    LinearGeneral: {
        "attn_dim": (50, 75),
        "num_heads": (1, 15)
    }
}


def is_seq_input(layer_type):
    return layer_type in [SelfAttention, EncoderBlock, LinearGeneral]


def random_generate_param_with_x(layer_type, seq_range=(175, 200), batch_size=1):
    if layer_type not in hyper_param_range:
        raise NotImplementedError(f"{layer_type} not supported")

    hyper_params = hyper_param_range[layer_type]
    hyper_params = {k: random.randint(s, e)
                    for k, (s, e) in hyper_params.items()}

    if layer_type is SelfAttention:
        hyper_params = {
            "heads": hyper_params["heads"],
            "in_dim": hyper_params['heads'] * hyper_params['attn_dim']
        }

    if layer_type is LinearGeneral:
        hyper_params = {
            "in_dim": (hyper_params['num_heads'] * hyper_params['attn_dim'], ),
            "feat_dim": (hyper_params['num_heads'], hyper_params['attn_dim'])
        }

    if layer_type is EncoderBlock:
        hyper_params = {
            "in_dim": hyper_params['num_heads'] * hyper_params['attn_dim'],
            "mlp_dim": hyper_params['mlp_dim'],
            "num_heads": hyper_params['num_heads'],
            "attn_dropout_rate": 0.
        }

    input_dim = hyper_params['in_dim'] if layer_type is not nn.Linear else hyper_params['in_features']

    input_dim = (input_dim, ) if type(input_dim) is int else input_dim
    input_dim = (random.randint(*seq_range),
                 *input_dim) if is_seq_input(layer_type) else (*input_dim, )

    return hyper_params, (batch_size, *input_dim)


def create_layer_from_param(layer_type, hyper_params, in_dim=None):
    return layer_type(**hyper_params), torch.randn(*in_dim) if in_dim else layer_type(**hyper_params)


def __test_layer():
    for k in hyper_param_range.keys():
        print(k)
        layer_hparam, in_dim = random_generate_param_with_x(k)
        layer, x = create_layer_from_param(k, layer_hparam, in_dim)
        layer.eval()
        layer(x)


def gen_dataset(layer_type, size=200):
    data = {}
    for i in range(size):
        hyparam, x = random_generate_param_with_x(layer_type)
        data[f"{i}"] = {
            "hyparam": hyparam,
            "in_dim": x
        }
    return data


def eval_dataset(path, layer_type):
    data = json.load(open(path, 'r'))
    label = {}
    for idx in tqdm(data, desc="Eval exection time"):
        hparam = data[idx]['hyparam']
        in_dim = data[idx]['in_dim']
        layer, x = create_layer_from_param(
            layer_type, hyper_params=hparam, in_dim=in_dim)
        label[idx] = timeit(layer, x)
    return label


def get_read_write_ops(layer_type, hyparam, in_dim):
    if layer_type is LinearGeneral:
        b, n, in_dim = in_dim
        return read, write, ops
    
    if layer_type is nn.Linear:
        b, in_dim = in_dim
        out_dim = hyparam['out_features']
        read = b * in_dim + in_dim * out_dim + out_dim + 2 * b *out_dim
        write = 2 * b * out_dim
        ops = b * out_dim * 2 * in_dim
        return read, write, ops

    if layer_type is SelfAttention:
        heads = hyparam['heads']
        head_dim = hyparam['in_dim'] // heads

if __name__ == '__main__':
    # __test_layer()
    # data = gen_dataset(LinearGeneral)
    # json.dump(data, open("lg-data.json", 'w'), indent=4)
    # result = eval_dataset("lg-data.json", LinearGeneral)
    # json.dump(result, open("lg-label.json", 'w'), indent=4)
    d = {"hyparam": {
            "in_features": 500,
            "out_features": 500
        },
        "in_dim": [
            1,
            219
        ]}
    get_read_write_ops(nn.Linear, **d)
    # LinearGeneral()(torch.randn(1, 197, 768))