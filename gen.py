import torch
import torch.nn as nn
from model import LinearGeneral, MlpBlock, SelfAttention, EncoderBlock
from timeIt import timeit
import random
import json
from tqdm import tqdm
import numpy as np

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
            "x_shape": x
        }
    return data

def eval_path(path, *args):
    return eval_dataset(json.load(open(path, 'r')), *args)


def eval_dataset(data, layer_type):
    label = {}
    for idx in tqdm(data, desc="Eval exection time"):
        hparam = data[idx]['hyparam']
        in_dim = data[idx]['x_shape']
        layer, x = create_layer_from_param(
            layer_type, hyper_params=hparam, in_dim=in_dim)
        label[idx] = timeit(layer, x)
    return label


def compute_memory_flops(layer_type, hyparam, x_shape):
    if layer_type is LinearGeneral:
        b, n, in_dim = x_shape
        feat_dim = hyparam['feat_dim']
        x_shape = (b, n, in_dim)

        x_read = np.prod(x_shape)
        weight_read = in_dim * np.prod(feat_dim)
        bias_read = np.prod(feat_dim)

        total_memory_read = x_read + weight_read + bias_read

        # Memory writes
        y_shape = (b, n, *feat_dim)
        total_memory_write = np.prod(y_shape)

        # FLOPs
        dot_product_flops = b * n * np.prod(feat_dim) * in_dim * 2
        addition_flops = np.prod(y_shape)
        total_flops = dot_product_flops + addition_flops

        return total_memory_read, total_memory_write, total_flops

    if layer_type is SelfAttention:
        b, n, in_dim = x_shape
        h = hyparam['heads']
        h_dim = in_dim // h
        read, write, flops = 0, 0, 0

        # x -> q, k, v
        q_read, q_write, q_flops = compute_memory_flops(LinearGeneral, {
            'in_dim': (in_dim, ),
            'feat_dim': (h, h_dim),
        }, (b, n, in_dim))
        q_shape = (b, h, n, h_dim)

        read += q_read * 3
        write += q_write * 3
        flops += q_flops * 3

        # q, k -> s
        read += np.prod(q_shape) * 2  # read q, k
        flops += b*h*n*n*h_dim*2  # q matmul k
        write += b*h*n*n

        # s * v -> z
        z_shape = (b, h, n, h_dim)
        read += np.prod(q_shape) + b*h*n*n
        flops += 2*b*h*n ^ 2*h_dim  # s matmul v
        write += np.prod(z_shape)
        out_shape = (b, h, in_dim)  # in_dim = h * h_dim

        # w_o(z) -> out
        out_read, out_write, out_flops = compute_memory_flops(LinearGeneral,
                                                              {'in_dim': (in_dim, ), 'feat_dim': (h, h_dim)}, (b, n, h_dim))
        read += out_read
        flops += out_flops
        write += out_write
        return read, write, flops

    if layer_type is nn.Linear:
        b, in_dim = x_shape
        out_dim = hyparam['out_features']
        read = b * in_dim + in_dim * out_dim + out_dim + 2 * b * out_dim
        write = 2 * b * out_dim
        ops = b * out_dim * 2 * in_dim
        return read, write, ops

    if layer_type is MlpBlock:
        b, in_dim = x_shape
        mlp_dim = hyparam['mlp_dim']
        out_dim = hyparam['out_dim']
        read, write, flops = 0, 0, 0

        # x -> fc1(x)
        fc1_r, fc1_w, fc1_flops = compute_memory_flops(nn.Linear, {
            'in_features': in_dim,
            'out_features': mlp_dim,
        }, (b, in_dim))
        read += fc1_r
        write += fc1_w
        flops += fc1_flops

        # x -> fc2(x)
        fc1_r, fc1_w, fc1_flops = compute_memory_flops(nn.Linear, {
            'in_features': mlp_dim,
            'out_features': out_dim,
        }, (b, in_dim))
        read += fc1_r
        write += fc1_w
        flops += fc1_flops
        return read, write, flops

    if layer_type is EncoderBlock:
        b, n, in_dim = x_shape
        mlp_dim = hyparam['mlp_dim']
        h = hyparam['num_heads']
        r, w, fp = 0, 0, 0

        # x -> x + msa(x)
        msa_r, msa_w, msa_fp = compute_memory_flops(SelfAttention, {
            'in_dim': in_dim, "heads": h
        }, (b, n, in_dim))
        r += msa_r
        w += msa_w
        fp += msa_fp

        # x -> x + ffn(x)
        ffn_r, ffn_w, ffn_fp = compute_memory_flops(MlpBlock, {
            'in_dim': in_dim, 'mlp_dim': mlp_dim, "out_dim": in_dim
        })
        r += ffn_r
        w += ffn_w
        fp += ffn_fp
        return r, w, fp


def estimate_exec_time(layer_type, param, speed, read_bdw, write_bdw, prec=32):
    '''
    speed #flop/s
    bdw xGB/s
    prec: int
    '''
    read, write, flops = compute_memory_flops(layer_type, **param)
    t_r = read * prec / read_bdw * 8e9
    t_w = write * prec / write_bdw * 8e9
    t_c = flops / speed 
    return t_r + t_w + t_c


def pred_path(path, *args):
    return pred_dataset(json.load(open(path, 'r')), *args)

def pred_dataset(data, layer_type, speed, read_bdw, write_bdw):
    pred = {}
    for idx in tqdm(data, desc="Estimate exection time"):
        pred[idx] = estimate_exec_time(layer_type, data[idx], speed, read_bdw, write_bdw)
    return pred

if __name__ == '__main__':
    # __test_layer()
    data = gen_dataset(LinearGeneral)
    json.dump(data, open("lg-data.json", 'w'), indent=4)
    
    pred = pred_dataset(data, LinearGeneral, 1000, 10, 10)
    json.dump(pred, open("lg-pred.json", 'w'), indent=4)
    
    result = eval_dataset(data, LinearGeneral)
    json.dump(result, open("lg-label.json", 'w'), indent=4)
