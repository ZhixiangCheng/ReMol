import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch import autograd
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from utils.public_utils import setup_device, Logger
from model.train_utils import fix_train_random_seed
from model.loss import calculate_loss, mse_loss, gather_features
from dataloader.pretrain_dataloader_lmdb2 import MoleculeDataset, my_collate_fn
from model.model import ReMol

# [*] Packages required to import distributed data parallelism
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# [*] Initialize the distributed process group and distributed device
def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)


def parse_args():
    parser = argparse.ArgumentParser(description='parameters of pretraining ReMol')

    # utils
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 2023)')
    parser.add_argument('--ckpt_dir', default='./ckpts/pretrain', help='path to checkpoint')
    parser.add_argument('--checkpoints', type=int, default=1,
                        help='how many iterations between two checkpoints (default: 1)')

    # train
    parser.add_argument('--nums', type=int, default=691826)
    parser.add_argument('--data_path', type=str, default="/data/chengzhixiang/ReMol/uspto_pretrain_fp", help='lmdb path')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--val_workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate (default: 5e-3)')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run (default: 50)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--temperature', type=float, default=0.05, help='temperature in contrastive loss')
    parser.add_argument('--RPCL_lambda', type=float, default=1,
                        help='start RPCL(Reactant-Product Contrastive Laerning) task, 1 means start, 0 means not start')
    parser.add_argument('--RRCL_lambda', type=float, default=1,
                        help='start RRCL(Reaction-Reaction Contrastive Learning) task, 1 means start, 0 means not start')
    parser.add_argument('--MRCI_lambda', type=float, default=1,
                        help='start MRCI(Mask Reaction Center Identification) task, 1 means start, 0 means not start')

    # DDP
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=2, type=int,
                        help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")

    return parser.parse_args()


def train(train_dataloader, model, criterion, optimizer, device, args):
    model.train()

    returnData = {
        "AvgTotalLoss": 0,
        "AvgRPCLLoss": 0,
        "AvgRRCLLoss": 0,
        "AvgMRCILoss": 0,
    }
    # with autograd.set_detect_anomaly(True):
    with tqdm(total=len(train_dataloader), position=0, ncols=120) as t:
        for i, batch_data in enumerate(train_dataloader):

            keys = ["reactant_img", "product_img", "label", "reactant_mask_atom", "reactant_mask_atom_label",
                    "reactant_mask_bond", "reactant_mask_bond_label", "product_mask_atom", "product_mask_atom_label",
                    "product_mask_bond", "product_mask_bond_label", "masks", "reaction_fp_sim"]

            reactant_img, product_img, label, reactant_mask_atom, reactant_mask_atom_label, reactant_mask_bond, \
                reactant_mask_bond_label, product_mask_atom, product_mask_atom_label, \
                product_mask_bond, product_mask_bond_label, masks, reaction_fp_sim = \
                [torch.autograd.Variable(batch_data[key].to(device)) for key in keys]

            reactant_embedding = model(reactant_img, masks, device, reactant=True, mlp=False)
            product_embedding = model(product_img, masks, device, reactant=False, mlp=False)

            reactant_embedding_bs = gather_features(reactant_embedding, args.world_size)
            product_embedding_bs = gather_features(product_embedding, args.world_size)
            reaction_fp_sim_bs = gather_features(reaction_fp_sim, args.world_size)

            # task1: RPCL (Reactant-Product Contrastive Learning)
            RPCL_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device)
            if args.RPCL_lambda != 0:
                RPCL_loss = calculate_loss(reactant_embedding_bs, product_embedding_bs, args.temperature)

            # task2: RRCL (Reaction-Reaction Contrastive Learning)
            RRCL_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device)
            if args.RRCL_lambda != 0:

                reaction_embedding = torch.cat((reactant_embedding_bs, product_embedding_bs), dim=1)
                RRCL_loss = mse_loss(reaction_embedding, reaction_fp_sim_bs)

            # task3: MRCI (Mask Reaction Center Identification)
            MRCI_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device)
            if args.MRCI_lambda != 0:
                reactant_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device)

                reactant_atom_embedding = model(reactant_mask_atom, masks, device, reactant=False, mlp=True)
                reactant_bond_embedding = model(reactant_mask_bond, masks, device, reactant=False, mlp=True)
                product_atom_embedding = model(product_mask_atom, masks, device, reactant=False, mlp=True)
                product_bond_embedding = model(product_mask_bond, masks, device, reactant=False, mlp=True)

                reactant_loss += criterion(reactant_atom_embedding,
                                           reactant_mask_atom_label)  # mask atom center identification
                reactant_loss += criterion(reactant_bond_embedding,
                                           reactant_mask_bond_label)  # mask bond center identification

                product_loss = torch.autograd.Variable(torch.Tensor([0.0])).to(device)

                product_loss += criterion(product_atom_embedding,
                                          product_mask_atom_label)  # mask atom center identification
                product_loss += criterion(product_bond_embedding,
                                          product_mask_bond_label)  # mask bond center identification

                MRCI_loss = reactant_loss + product_loss

            # calculating all loss to backward
            loss = RPCL_loss * args.RPCL_lambda + RRCL_loss * args.RRCL_lambda + MRCI_loss * args.MRCI_lambda


            # calculating average loss
            returnData["AvgRPCLLoss"] += RPCL_loss.item() / len(train_dataloader)
            returnData["AvgRRCLLoss"] += RRCL_loss.item() / len(train_dataloader)
            returnData["AvgMRCILoss"] += MRCI_loss.item() / len(train_dataloader)
            returnData["AvgTotalLoss"] += loss.item() / len(train_dataloader)

            # compute gradient and do SGD step
            if loss.item() != 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if dist.get_rank() == 0:
                t.set_postfix(TotalLoss=loss.item(), RPCL_Loss=RPCL_loss.item(), RRCL_Loss=RRCL_loss.item(),
                              MRCI_Loss=MRCI_loss.item())
                t.update(1)

    return returnData


def eval(val_dataloader, model, device, args):
    returnData = {
        "ReactionAcc": 0,
        "MRCIAcc": 0,
        "ReactantAtomAcc": 0,
        "ReactantBondAcc": 0,
        "ProductAtomAcc": 0,
        "ProductBondAcc": 0,
    }

    # evaluation
    with torch.no_grad():
        reactant_atom_correct = 0
        reactant_bond_correct = 0
        product_atom_correct = 0
        product_bond_correct = 0
        # reaction_correct = 0

        reactant_num = 0
        product_num = 0
        # reaction_num = 0

        for batch_data in tqdm(val_dataloader, total=len(val_dataloader), position=0, ncols=120):
            keys = ["reactant_img", "product_img", "label", "reactant_mask_atom", "reactant_mask_atom_label",
                    "reactant_mask_bond", "reactant_mask_bond_label", "product_mask_atom", "product_mask_atom_label",
                    "product_mask_bond", "product_mask_bond_label", "masks"]

            reactant_img, product_img, label, reactant_mask_atom, reactant_mask_atom_label, reactant_mask_bond, \
                reactant_mask_bond_label, product_mask_atom, product_mask_atom_label, \
                product_mask_bond, product_mask_bond_label, masks = \
                [torch.autograd.Variable(batch_data[key].to(device)) for key in keys]

            reactant_atom_embedding = model(reactant_mask_atom, mask=[], device=device, reactant=False, mlp=True)
            reactant_bond_embedding = model(reactant_mask_bond, mask=[], device=device, reactant=False, mlp=True)
            product_atom_embedding = model(product_mask_atom, mask=[], device=device, reactant=False, mlp=True)
            product_bond_embedding = model(product_mask_bond, mask=[], device=device, reactant=False, mlp=True)

            _, reactant_atom_pred = reactant_atom_embedding.max(dim=1)
            _, reactant_bond_pred = reactant_bond_embedding.max(dim=1)
            _, product_atom_pred = product_atom_embedding.max(dim=1)
            _, product_bond_pred = product_bond_embedding.max(dim=1)

            reactant_atom_correct += torch.sum(reactant_atom_pred == reactant_mask_atom_label)
            reactant_bond_correct += torch.sum(reactant_bond_pred == reactant_mask_bond_label)
            product_atom_correct += torch.sum(product_atom_pred == product_mask_atom_label)
            product_bond_correct += torch.sum(product_bond_pred == product_mask_bond_label)

            reactant_num += reactant_mask_atom_label.shape[0]
            product_num += product_mask_atom_label.shape[0]

        returnData["ReactantAtomAcc"] = float(reactant_atom_correct) / reactant_num
        returnData["ReactantBondAcc"] = float(reactant_bond_correct) / reactant_num
        returnData["ProductAtomAcc"] = float(product_atom_correct) / product_num
        returnData["ProductBondAcc"] = float(product_bond_correct) / product_num
        returnData["MRCIAcc"] = (returnData["ReactantAtomAcc"] + returnData["ReactantBondAcc"] + returnData[
            "ProductAtomAcc"] + returnData["ProductBondAcc"]) / 4

    return returnData


def main(local_rank, ngpus_per_node, args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(args.log_dir + "pretrain.log")

    tb_writer = SummaryWriter()

    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

    # fix random seeds
    fix_train_random_seed(args.seed)

    transform = [transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]), ]

    train_index, val_index = train_test_split(list(range(args.nums)), test_size=0.2, shuffle=True, random_state=2023)

    train_dataset = MoleculeDataset(train_index, args.data_path, transforms.Compose(transform))
    val_dataset = MoleculeDataset(val_index, args.data_path, transforms.Compose(transform))

    batch_size = args.batch // args.world_size  # [*] // world_size

    train_sampler = DistributedSampler(train_dataset, shuffle=True)  # [*]
    val_sampler = DistributedSampler(val_dataset, shuffle=False)  # [*]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=args.workers, pin_memory=True, collate_fn=my_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=args.val_workers, pin_memory=True,
                                                 collate_fn=my_collate_fn)

    model = ReMol(dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                                  strict=False)
            print("=> loading completed")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = model.to(device)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
                broadcast_buffers=False)  # [*] DDP(...)

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(args.start_epoch, args.epochs):

        # [*] set sampler
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)


        # train
        TrainData = train(train_dataloader, model, criterion, optimizer, device, args)

        scheduler.step()

        # eval
        model.eval()
        evaluationData = eval(val_dataloader, model, device, args)

        # [*] save model on rank 0
        if dist.get_rank() == 0:
            # save model
            saveRoot = os.path.join(args.ckpt_dir, 'checkpoints')
            if not os.path.exists(saveRoot):
                os.makedirs(saveRoot)
            if epoch % args.checkpoints == 0:
                saveFile = os.path.join(saveRoot, 'ReMol_{}.pth.tar'.format(epoch + 1))
                if args.verbose:
                    print_only_rank0('Save checkpoint at: {}'.format(saveFile))

                if isinstance(model, torch.nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch.save({
                    'state_dict': model_state_dict,
                }, saveFile)

            print_only_rank0('Epoch: [{}][train]\t'
                             'TotalLoss：{:.3f}\t'
                             'RPCLLoss：{:.3f}\t'
                             'RRCLLoss：{:.3f}\t'
                             'MRCILoss：{:.3f}\t'
                             .format(epoch + 1, TrainData["AvgTotalLoss"], TrainData["AvgRPCLLoss"],
                                     TrainData["AvgRRCLLoss"],
                                     TrainData["AvgMRCILoss"]))

            print_only_rank0('Epoch: [{}][val]\t'
                             'ReactantAtomAcc：{:.3f}\t'
                             'ReactantBondAcc：{:.3f}\t'
                             'ProductAtomAcc：{:.3f}\t'
                             'ProductBondAcc：{:.3f}\t\n'
                             .format(epoch + 1, evaluationData['ReactantAtomAcc'],
                                     evaluationData['ReactantBondAcc'],
                                     evaluationData["ProductAtomAcc"], evaluationData["ProductBondAcc"]))

            tags = ["TotalLoss", "RPCLLoss", "RRCLLoss", "MRCILoss", "MRCIAcc"]

            tb_writer.add_scalar(tags[0], TrainData["AvgTotalLoss"], epoch + 1)
            tb_writer.add_scalar(tags[1], TrainData["AvgRPCLLoss"], epoch + 1)
            tb_writer.add_scalar(tags[2], TrainData["AvgRRCLLoss"], epoch + 1)
            tb_writer.add_scalar(tags[3], TrainData["AvgMRCILoss"], epoch + 1)
            tb_writer.add_scalar(tags[4], evaluationData['MRCIAcc'], epoch + 1)

    print_only_rank0("used time: {}".format(time.time() - start_time))
    sys.stdout.close()


if __name__ == '__main__':
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes

    # [*] run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
