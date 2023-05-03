from mmsa import mix_prune, get_mask_idx
from quant_utils import set_mixed_precision, get_single_prec_quant_strategy
from model import get_vit
import args
import torch
import torch.nn as nn
import numpy as np
from env import QuantPruneEnv
from data import get_cifar10_dataloader
from train import finetune, eval_model


def eval_unipi(model, trainloader, testloader, device, w_bit, heads, a_bit=8):

    pr_idx = get_mask_idx(model)
    pr_s = [heads] * len(pr_idx)

    q_idx, q_s = get_single_prec_quant_strategy(model, w_bit, a=a_bit)

    lat, e, mem = QuantPruneEnv.estimate_strategy([w_bit]*48, pr_s, args.COEF_LAT, args.COEF_E)

    set_mixed_precision(model, q_idx, q_s)
    mix_prune(model, pr_idx, pr_s)


    finetune(model, trainloader, device)
    acc = eval_model(model, testloader, device)

    return acc, lat, e, mem


if __name__ == '__main__':
    trainloader = get_cifar10_dataloader(train=True)
    testloader = get_cifar10_dataloader(train=False)
    device = 'npu'

    w = [6]
    h = [3, 4, 5, 6]

    with open('eval-uni.csv', 'a+') as f:
        for w_b in reversed(w):
            for heads in reversed(h):
                model = get_vit(args.MQVIT, r'/home/ma-user/work/Vision-Transformer-ViT/output/mvit/pat-0.5.pt')
                acc, lat, e, mem = eval_unipi(model ,trainloader, testloader, device, w_b, heads)
                f.write(f'{w_b},{heads},{acc},{lat},{e},{mem}\n')
                f.flush()

