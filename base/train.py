# -*- coding: utf-8 -*-
# @File : train.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:11
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import CAFIA_Transformer
from mmsa import build_mvit, get_masks_from_mmsa, freeze_model_but_mask, get_mask_val_from_masks
from tqdm import tqdm
import logging
from scheduler import cosine_lr
from data import get_cifar10_dataloader
from torch.utils.tensorboard import SummaryWriter

import torch_npu

logging.basicConfig(level = logging.NOTSET)
device = torch.device("npu")
 
def train_model(args, trainloader, testloader):
    n_gpu = 1
    model = CAFIA_Transformer(args)
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), betas=(args.beta1, args.beta2), eps = args.eps, lr = args.learning_rate, weight_decay = args.weight_decay)
    
    total_steps = (len(trainloader) // args.batch_size + 1)  * args.epoches
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)   
    nb_tr_steps = 0
    logging.info('**************************** start to train *******************************')
    for epoch in range(args.epoches):
        train_loss = 0 
        train_iter = 0
        for _, batch in enumerate(tqdm(trainloader, desc = "Iteration")):
            nb_tr_steps += 1  
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            scheduler(nb_tr_steps)
            train_loss += loss.item()
            train_iter += 1
            logging.info('Epoch:%d batch_loss:%f', epoch, loss)
        
        train_loss = loss / train_iter

        #eval
        logging.info('**************************** start to evaluate *******************************')
        model.eval()
        total, correct = 0, 0
        for _, batch in enumerate(tqdm(testloader, desc = "Iteration")):
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum()
            
        acc = (correct / total).item()
        logging.info('Epoch: %d train_loss: %f Accuracy: %f', epoch, train_loss, acc)
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        output_path = os.path.join(args.output, str(acc)+'.pt')
        torch.save(model.state_dict(), output_path)


def eval_model(model, dataloader, device='cpu'):
    logging.info('**************************** start to evaluate *******************************')
    model.eval()
    model = model.to(device)
    total, correct = 0, 0
    for _, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
        batch = tuple(t.to(device) for t in batch)
        batch_X, batch_Y = batch
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_Y.size(0)
        correct += (predicted == batch_Y).sum()
        
    acc = (correct / total).item()
    logging.info('Accuracy: %f', acc)


def loss_fn(x, target, model, alpha=1e-2):
    loss = F.cross_entropy(x, target)
    # masks = get_masks_from_mmsa(model)
    masks = get_mask_val_from_masks(model)
    for mask in masks:
        loss += torch.norm(mask, 2) * alpha
    return loss


def train_mvit(args, trainloader=None, model=None, freeze_w=False):
    
    if not model:
        model = build_mvit(args)
        model = model.to(device)

    if freeze_w:
        model = freeze_model_but_mask(model)
    
    if not trainloader:
        trainloader = get_cifar10_dataloader(root=os.path.join(args.pwd, './data'),train=True, batch_size=args.batch_size)
    
    model.train()

    pr = model.vit.transformer.encoder_layers[0].attn.mask.prune_rate

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    alpha = args.alpha
    total_steps = (len(trainloader) // args.batch_size + 1)  * args.epoches
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)   
    nb_tr_steps = 0

    def print_mask_grad():
        for mask in get_masks_from_mmsa(model):
            print(mask.grad)

    writer = SummaryWriter(comment=f'-prune_rate={pr}')

    logging.info('**************************** start to train mvit *******************************')
    for epoch in range(args.epoches):
        train_loss = 0 
        train_iter = 0
        for _, batch in enumerate(tqdm(trainloader, desc = "Iteration")):
            nb_tr_steps += 1  
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y, model, alpha)
            loss.backward()
            # print_mask_grad()

            mask = model.vit.transformer.encoder_layers[0].attn.mask.mask
            grad = mask.grad

            writer.add_scalars('mask_val', {f'mask0-{i}': mask[i] for i in range(len(mask))}, nb_tr_steps)
            # writer.add_scalars('mask_grad', {f'grad0-{i}': grad[i] for i in range(len(grad))}, idx)

            optimizer.step()
            scheduler(nb_tr_steps)
            train_loss += loss.item()
            train_iter += 1
            logging.info('Epoch:%d batch_loss:%f', epoch, loss)
        
        train_loss = loss / train_iter

        #eval
        # logging.info('**************************** start to evaluate *******************************')
        # model.eval()
        # total, correct = 0, 0
        # for _, batch in enumerate(tqdm(testloader, desc = "Iteration")):
        #     batch = tuple(t.to(device) for t in batch)
        #     batch_X, batch_Y = batch
        #     outputs = model(batch_X)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += batch_Y.size(0)
        #     correct += (predicted == batch_Y).sum()
            
        # acc = (correct / total).item()
        # logging.info('Epoch: %d train_loss: %f Accuracy: %f', epoch, train_loss, acc)
        if epoch % 10 == 0:
            dir_path = os.path.join(args.pwd, args.output, f'pr={pr}')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            output_path = os.path.join(dir_path, str(epoch//10)+'.pt')
            torch.save(model.state_dict(), output_path)

    return model

if __name__ == '__main__':
    import json
    from argparse import Namespace
    from data import get_cifar10_dataloader
    train_loader = get_cifar10_dataloader()
    test_loader = get_cifar10_dataloader(train=False)

    pwd = '/home/ma-user/work/Vision-Transformer-ViT/ViT-B_16-224.json'

    args = Namespace(**json.load(open(pwd, 'r')))

    train_mvit(args, train_loader, freeze_w=True)

    # args.cifar10_vit = './output/mvit/2023-03-13-6.pt'
    # mvit = build_mvit(args)

    # eval_model(mvit, test_loader, device)

