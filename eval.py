import torch
import torch.nn as nn
from model import*
from quant_utils import *
from train import eval_model, finetune
from data import get_cifar10_dataloader
import args

def test_case(vit, w, a, hwq:bool, device, need_finetune=False, trainloader=None):
    set_hwq(vit, hwq)
    idx, strategy = get_single_prec_quant_strategy(vit, w, a)
    set_mixed_precision(vit, idx, strategy)
    if need_finetune:
        finetune(vit, trainloader, device)
    return eval_model(vit, test_loader, device)


if __name__ == '__main__':

    import torch_npu
    device = torch.device("npu")

    path = '/home/ma-user/work/Vision-Transformer-ViT/output/vit/0.9853000044822693.pt'

    res_path = '/home/ma-user/work/design-code/result.txt'

    train_loader = get_cifar10_dataloader()
    test_loader = get_cifar10_dataloader(train=False)

    W = [2, 3, 4, 5, 6, 7, 8]
    A = [4, 8]
    Hwq = [True, False]

    with open(res_path, 'w') as f:
        for a in A[::-1]:
            for w in W[::-1]:
                for hwq in Hwq:
                    vit = get_vit(args.QVIT, path)
                    accur = test_case(vit, w, a, hwq=hwq, device=device, need_finetune=False)
                    f.write(f'w: {w}, a: {a}, hwq: {hwq}, without finetune: {accur}')
                    f.flush()
                    if accur < 0.9:
                        accur = test_case(vit, w, a, hwq=hwq, device=device, need_finetune=True, trainloader=train_loader)
                        f.write(f'w: {w}, a: {a}, hwq: {hwq}, finetuned: {accur}')
                        f.flush()