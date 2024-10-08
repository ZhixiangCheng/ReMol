# -*- coding: UTF-8 -*-
"""
Project -> File: molecule -> finetuning
Author: cer0
Date: 2023-04-18 21:01
Description:

"""
import argparse
import os
import sys
import torchvision.transforms as transforms
import torchvision
import torch
from tqdm import tqdm
import numpy as np
import timm
import torch.nn as nn
from sklearn import metrics
import logging
import pandas as pd
from sklearn import model_selection
from torch.optim import Adam
from model.train_utils import PolynomialDecayLR

from utils.public_utils import setup_device, Logger, cal_torch_model_params, is_left_better_right
from model.train_utils import ReMol, fix_train_random_seed, save_finetune_ckpt, metric_reg, calc_cliff_rmse
from dataloader.image_dataloader import ImageDataset, train_test_val_idx
from model.train_utils import metric as utils_evaluate_metric
from model.train_utils import metric_multitask as utils_evaluate_metric_multitask
from rdkit import RDLogger

# 禁用RDKit的警告日志
RDLogger.DisableLog('rdApp.warning')


def train_one_epoch(model, optimizer, data_loader, criterion, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        sample_num += images.shape[0]

        pred = model(images)

        # print(pred.shape)
        # print(labels.shape)
        labels = labels.view(pred.shape).to(torch.float64)

        is_valid = labels != -1
        loss_mat = criterion(pred.double(), labels)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        # if not torch.isfinite(loss):
        #     print('WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, epoch, return_data_dict=False):
    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        with torch.no_grad():
            pred = model(images)
            # print(pred.shape)
            labels = labels.view(pred.shape).to(torch.float64)
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)

            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        y_true.append(labels)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        y_pro = torch.sigmoid(torch.Tensor(y_scores))
        y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
        if return_data_dict:
            data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
            return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
        else:
            return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)

    elif y_true.shape[1] > 1:
        y_pro = torch.sigmoid(torch.Tensor(y_scores))
        y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
        # print(y_true.shape, y_pred.shape, y_pro.shape)
        if return_data_dict:
            data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
            return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro,
                                                                                  num_tasks=y_true.shape[1],
                                                                                  empty=-1), data_dict
        else:
            return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro,
                                                                                  num_tasks=y_true.shape[1], empty=-1)


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sys.stdout = Logger(
        args.log_dir + "finetuning_{}_{}_{}_{}_{}.log".format(args.dataset, args.resume, args.eval_metric, args.runseed,
                                                              args.lr))

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    ##################################### load data #####################################
    if args.image_aug:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                 transforms.ToTensor()]
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_smi, train_label, val_smi, val_label, test_smi, test_label = train_test_val_idx(args.dataroot, args.dataset,
                                                                                          args.split_name)

    num_tasks = train_label.shape[1]

    train_dataset = ImageDataset(train_smi, train_label, img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize)
    val_dataset = ImageDataset(val_smi, val_label, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize)
    test_dataset = ImageDataset(test_smi, test_label, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################
    model = ReMol(dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=num_tasks)

    resume = "./ckpts/pretrain/checkpoints/" + args.resume + ".pth.tar"

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            del_keys = ['module.mlp_head.weight', 'module.mlp_head.bias']
            for k in del_keys:
                del checkpoint['state_dict'][k]
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                                  strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    ##################################### initialize optimizer #####################################
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none").cuda()

    eval_metric = args.eval_metric
    valid_select = "max"
    min_value = -np.inf

    ##################################### train #####################################
    # min_value = np.inf
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None}

    early_stop = 0
    patience = 30

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                        device=device, epoch=epoch)

        # evaluate
        train_loss, train_results, train_data_dict = evaluate(model=model, data_loader=train_dataloader,
                                                               criterion=criterion, device=device,
                                                               epoch=epoch, return_data_dict=True)

        val_loss, val_results, val_data_dict = evaluate(model=model, data_loader=val_dataloader,
                                                         criterion=criterion, device=device,
                                                         epoch=epoch, return_data_dict=True)

        test_loss, test_results, test_data_dict = evaluate(model=model, data_loader=test_dataloader,
                                                            criterion=criterion, device=device,
                                                            epoch=epoch, return_data_dict=True)

        scheduler.step()

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['final_train'] = train_result
            results['highest_valid'] = valid_result
            results['final_test'] = test_result

            results['final_train_desc'] = train_results
            results['highest_valid_desc'] = val_results
            results['final_test_desc'] = test_results

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(args.resume, model, optimizer, round(train_loss, 4), epoch, args.ckpt_dir,
                                   args.dataset, lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("final results: highest_valid: {}, final_train: {}, final_test: {}"
          .format(results["highest_valid"], results["final_train"], results["final_test"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ReMol')

    # basic
    parser.add_argument('--dataset', type=str, default="bbbp", help='dataset name, e.g. bace, bbbp, ...')
    parser.add_argument('--dataroot', type=str, default="./datasets/finetuning/molenet/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=5, type=int, help='number of data loading workers (default: 5)')
    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-6, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=0, help='random seed to run model (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='ReMol', type=str, metavar='PATH')
    parser.add_argument('--split_name', default='0', type=str, metavar='PATH', help='')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_aug', action='store_true', default=True, help='whether to use data augmentation')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--eval_metric', type=str, default="auroc", help='eval metric')
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--ckpt_dir', default='./ckpts/finetuning', help='path to checkpoint')

    args = parser.parse_args()
    main(args)
