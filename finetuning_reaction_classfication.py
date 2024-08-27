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
import torch
from tqdm import tqdm
import numpy as np

from utils.public_utils import setup_device, Logger, is_left_better_right
from model.train_utils import ReMol_Reaction, fix_train_random_seed, save_finetune_ckpt
from dataloader.image_dataloader import ReactionDataset, reaction_class_pred, my_collate_fn
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
        keys = ["reactant_img", "product_img", "label", "masks"]

        reactant_img, product_img, labels, masks = \
            [data[key].to(device) for key in keys]

        sample_num += product_img.shape[0]

        pred = model(reactant_img, product_img, masks, device)

        is_valid = labels != -1
        loss_mat = criterion(pred.double(), labels)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        loss.backward()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, epoch, return_data_dict=False):
    model.eval()

    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    correct = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        keys = ["reactant_img", "product_img", "label", "masks"]

        reactant_img, product_img, labels, masks = \
            [data[key].to(device) for key in keys]

        sample_num += product_img.shape[0]

        with torch.no_grad():
            pred = model(reactant_img, product_img, masks, device)
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)

            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

            _, reaction_pred = pred.max(dim=1)
            correct += torch.sum(reaction_pred == labels)

    acc = float(correct)/float(sample_num)

    return accu_loss.item() / (step + 1), acc



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

    train_reactant, train_product, train_label, val_reactant, val_product, val_label, test_reactant, test_product, test_label = reaction_class_pred(
        args.dataroot, args.k, args.runseed)

    num_tasks = 46

    train_dataset = ReactionDataset(train_reactant, train_product, train_label,
                                    img_transformer=transforms.Compose(img_transformer_train),
                                    normalize=normalize)
    val_dataset = ReactionDataset(val_reactant, val_product, val_label,
                                  img_transformer=transforms.Compose(img_transformer_test),
                                  normalize=normalize)
    test_dataset = ReactionDataset(test_reactant, test_product, test_label,
                                   img_transformer=transforms.Compose(img_transformer_test),
                                   normalize=normalize)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   collate_fn=my_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 collate_fn=my_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True,
                                                  collate_fn=my_collate_fn)

    ##################################### load model #####################################
    model = ReMol_Reaction(dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=num_tasks)

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

    criterion = torch.nn.CrossEntropyLoss().cuda()

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
        train_loss, train_result = evaluate(model=model, data_loader=train_dataloader,
                                                               criterion=criterion, device=device,
                                                               epoch=epoch, return_data_dict=True)

        val_loss, valid_result = evaluate(model=model, data_loader=val_dataloader,
                                                         criterion=criterion, device=device,
                                                         epoch=epoch, return_data_dict=True)

        test_loss, test_result = evaluate(model=model, data_loader=test_dataloader,
                                                            criterion=criterion, device=device,
                                                            epoch=epoch, return_data_dict=True)

        scheduler.step()


        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['final_train'] = train_result
            results['highest_valid'] = valid_result
            results['final_test'] = test_result

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
    parser.add_argument('--dataroot', type=str, default="./datasets/finetuning/schneider/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 2)')
    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-6, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=0, help='random seed to run model (default: 0)')
    parser.add_argument('--k', type=int, default=4, help='sapmle num')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=1, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='ReMol', type=str, metavar='PATH',
                        help='./ckpts/pretrain/checkpoints/ReMol.pth.tar')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_aug', action='store_true', default=True, help='whether to use data augmentation')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--ckpt_dir', default='./ckpts/finetuning', help='path to checkpoint')

    args = parser.parse_args()
    main(args)
