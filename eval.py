import torch
import torch.nn as nn
from model import *
from quant_utils import *
from train import eval_model, finetune, train_mvit
from data import get_cifar10_dataloader
import args
from mmsa import *


def test_case(vit, w, a, hwq: bool, device, test_loader, need_finetune=False, trainloader=None):
    set_hwq(vit, hwq)
    idx, strategy = get_single_prec_quant_strategy(vit, w, a)
    set_mixed_precision(vit, idx, strategy)
    if need_finetune:
        finetune(vit, trainloader, device)
    return eval_model(vit, test_loader, device)


def eval_hwq(device, weight_path, output_path):

    train_loader = get_cifar10_dataloader()
    test_loader = get_cifar10_dataloader(train=False)

    W = [2, 3, 4, 5, 6, 7, 8]
    A = [-1]
    # A = [4, 8]
    Hwq = [True, False]

    with open(output_path, 'w') as f:
        for a in A[::-1]:
            for w in W[::-1]:
                for hwq in Hwq:
                    vit = get_vit(args.QVIT, weight_path)
                    accur = test_case(
                        vit, w, a, test_loader=test_loader, hwq=hwq, device=device, need_finetune=False)
                    f.write(
                        f'w: {w}, a: {a}, hwq: {hwq}, without finetune, {accur}\n')
                    f.flush()
                    if accur < 0.9:
                        accur = test_case(vit, w, a, test_loader=test_loader, hwq=hwq,
                                          device=device, need_finetune=True, trainloader=train_loader)
                        f.write(
                            f'w: {w}, a: {a}, hwq: {hwq}, finetuned, {accur}\n')
                        f.flush()


def get_mvit(path, args):
    args.cifar10_vit = path
    mvit = build_mvit(args)
    return mvit


def eval_tmp_pat(train_loader, test_loader, args, output_path, device):
    ws = {
        'pat-0.75': './output/mvit/pat-0.75.pt',
        'pat-0.5':'./output/mvit/pat-0.5.pt',
        'tmp': './output/mvit/tmp.pt'
    }

    test_pr = [0.75, 0.5, 0.25]
    with open(output_path, 'w') as f:
        for pat, path in ws.items():
            for pr in test_pr:
                mvit = get_mvit(path, args)
                prune(mvit, pr)
                accur = eval_model(mvit, test_loader, device)
                f.write(f'{pat} sparsity-{pr} accur(no finetune): {accur}\n')
                f.flush()
                if pr < 0.6:
                    args.epoches = 1
                    args.pr = 1.
                    train_mvit(args, trainloader=train_loader, model=mvit, device=device,
                               save_path=f'./Vision-Transformer-ViT/output/mvit/{pat}-pr-{pr}.pt', log_info=False)
                    accur = eval_model(mvit, test_loader, device)
                    f.write(f'{pat} sparsity-{pr} accur(finetuned): {accur}\n')
                    f.flush()

        
        path = ws['tmp'] # test random head pruning
        for pr in test_pr:
            mvit = get_mvit(path, args)
            prune(mvit, pr, random=True)
            show_dmask(mvit)
            accur = eval_model(mvit, test_loader, device)
            f.write(
                f'random pruning sparsity-{pr} accur(no finetune): {accur}\n')
            f.flush()
            if pr < 0.6:
                args.epoches = 1
                args.pr = 1.
                fix_dmask(mvit)
                train_mvit(args, trainloader=train_loader, model=mvit, device=device,
                           log_info=False, save_path=f'Vision-Transformer-ViT/output/mvit/rand-{pr}.pt')
                show_dmask(mvit)
                accur = eval_model(mvit, test_loader, device)
                f.write(
                    f'random pruning sparsity-{pr} accur(finetuned): {accur}\n')
                f.flush()

if __name__ == '__main__':
#     path = '/Users/qing/Downloads/pat-0.5.pt'
#     vit = get_vit(args.MQVIT, path)
#     print(vit)
#     print(vit(torch.randn(1, 3, 224, 224)).shape)
    pass