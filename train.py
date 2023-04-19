# -*- coding: utf-8 -*-
# @File : train.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:11
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import CAFIA_Transformer
from tqdm import tqdm
import logging
from scheduler import cosine_lr
import torch.nn.functional as F
from mmsa import build_mvit, freeze_model_but_mask, get_mask_val_from_masks, set_mask_prune_rate, to_mask, fix_dmask
from data import get_cifar10_dataloader

logging.basicConfig(level=logging.NOTSET)


def train_model(args, trainloader, testloader, device='cpu'):
    n_gpu = 1
    model = CAFIA_Transformer(args)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), betas=(
        args.beta1, args.beta2), eps=args.eps, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = (len(trainloader) // args.batch_size + 1) * args.epoches
    scheduler = cosine_lr(optimizer, args.learning_rate,
                          args.warmup, total_steps)
    nb_tr_steps = 0
    logging.info(
        '**************************** start to train *******************************')
    for epoch in range(args.epoches):
        train_loss = 0
        train_iter = 0
        for _, batch in enumerate(tqdm(trainloader, desc="Iteration")):
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

        train_loss /= train_iter

        # eval
        logging.info(
            '**************************** start to evaluate *******************************')
        model.eval()
        total, correct = 0, 0
        for _, batch in enumerate(tqdm(testloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum()

        acc = (correct / total).item()  # type: ignore
        logging.info('Epoch: %d train_loss: %f Accuracy: %f',
                     epoch, train_loss, acc)
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        output_path = os.path.join(args.output, str(acc)+'.pt')
        torch.save(model.state_dict(), output_path)


def eval_model(model, dataloader, device='cpu'):
    logging.info(
        '**************************** start to evaluate *******************************')
    model.eval()
    model = model.to(device)
    total, correct = 0, 0
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        batch_X, batch_Y = batch
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_Y.size(0)
        correct += (predicted == batch_Y).sum()

    acc = (correct / total).item()  # type: ignore
    logging.info('Accuracy: %f', acc)
    return acc


def finetune(model: nn.Module, trainloader, device, epoches=1, save_dir=None, fname=None, **optim_kwargs):

    logging.info(
        f'**************************** start to finetune {epoches} turns*******************************')
    model.train()
    model = model.to(device)

    if 'lr' not in optim_kwargs:
        optim_kwargs['lr'] = 1e-4

    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        for _, batch in enumerate(tqdm(trainloader, desc="Iteration")):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            logging.info('Epoch:%d batch_loss:%f', epoch, loss)

    if save_dir and fname:
        if not os.path.exists(save_dir):
            pass
        torch.save(model.state_dict(), os.path.join(save_dir, fname+'.pt'))
    return model


def train_mvit(args, trainloader=None, model=None, freeze_w=False, device='cpu', save_path=None, log_info=True):

    def loss_fn(x, target, model, alpha=1e-2):
        loss = F.cross_entropy(x, target)
        # masks = get_masks_from_mmsa(model)
        masks = get_mask_val_from_masks(model)
        for mask in masks:
            loss += torch.norm(mask, 2) * alpha  # type: ignore
        return loss

    if not model:
        model = build_mvit(args)
    model = model.to(device)
    to_mask(model, device)

    if freeze_w:
        model = freeze_model_but_mask(model)

    fix_dmask(model)

    if not trainloader:
        trainloader = get_cifar10_dataloader(root=os.path.join(
            args.pwd, './data'), train=True, batch_size=args.batch_size)

    model.train()

    pr = args.pr if hasattr(args, 'pr') else 1.0
    set_mask_prune_rate(model, pr)  # type: ignore

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    alpha = args.alpha
    # total_steps = (len(trainloader) // args.batch_size + 1)  * args.epoches
    # scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
    nb_tr_steps = 0

    logging.info(
        '**************************** start to train mvit *******************************')
    for epoch in range(args.epoches):
        train_loss = 0
        train_iter = 0
        for _, batch in enumerate(tqdm(trainloader, desc="Iteration")):
            nb_tr_steps += 1
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            batch_X, batch_Y = batch
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_Y, model, alpha)
            loss.backward()
            # print_mask_grad()

            if log_info:
                pass
                # mask = model.vit.transformer.encoder_layers[0].attn.mask.get_mask()
                # writer.add_scalars('masks', {f'mask-{i}': mask[i] for i in range(len(mask))}, nb_tr_steps)
                # stds = [torch.std(m) for m in get_mask_val_from_masks(model)]
                # writer.add_scalars('stds', {f'std-{i}': stds[i] for i in range(len(stds))}, nb_tr_steps)

            optimizer.step()
            # scheduler(nb_tr_steps)
            train_loss += loss.item()
            train_iter += 1
            # logging.info('Epoch:%d batch_loss:%f', epoch, loss)

        train_loss = train_loss / train_iter

        if epoch % 10 == 0:
            dir_path = os.path.join(args.pwd, args.output, f'pr={pr}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            output_path = os.path.join(dir_path, str(
                train_loss)+'.pt') if save_path is None else save_path
            logging.info(
                f'**************************** save model to {output_path} *******************************')
            torch.save(model.state_dict(), output_path)
    return model
