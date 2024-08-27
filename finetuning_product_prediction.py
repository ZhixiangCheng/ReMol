import argparse
import os
import sys
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


from utils.public_utils import setup_device, Logger
from model.train_utils import ReMol_Product, fix_train_random_seed
from dataloader.image_dataloader import ReactionDataset, my_collate_fn
from rdkit import RDLogger


# 禁用RDKit的警告日志
RDLogger.DisableLog('rdApp.warning')


def evaluate(args, model, dataset, device):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        all_rankings = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True, collate_fn=my_collate_fn)
        for data in tqdm(dataloader):

            product_img = data["product_img"].to(device)
            masks = data["masks"].to(device)

            product_embeddings = model(product_img, masks, device, reactant=False)
            all_product_embeddings.append(product_embeddings)

        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        i = 0
        for data in tqdm(dataloader):
            keys = ["reactant_img", "masks"]

            reactant_img, masks = [data[key].to(device) for key in keys]

            reactant_embeddings = model(reactant_img, masks, device, reactant=True)

            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch, len(dataset))), dim=1)
            i += args.batch
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        np.save('ReMol_ranking.npy', all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        print('mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mrr, mr, h1, h3, h5, h10))
        return mrr




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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data = pd.read_csv(args.dataroot + '{}.csv'.format(args.dataset))
    reactant = data["reactant"].tolist()
    product = data["product"].tolist()
    label = [0]*len(reactant)

    dataset = ReactionDataset(reactant, product, label, img_transformer=transforms.Compose(img_transformer_train),
                              normalize=normalize)

    ##################################### load model #####################################
    model = ReMol_Product(dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=1)

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

    evaluate(args, model, dataset, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ReMol')

    # basic
    parser.add_argument('--dataroot', type=str, default="./datasets/finetuning/USPTO-479k/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=5, type=int, help='number of data loading workers (default: 2)')
    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=0, help='random seed to run model (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=1, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='ReMol', type=str, metavar='PATH')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_aug', action='store_true', default=True, help='whether to use data augmentation')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    parser.add_argument('--log_dir', type=str, default="./logs/", help='log dir')
    parser.add_argument('--ckpt_dir', default='./ckpts/finetuning', help='path to checkpoint')

    args = parser.parse_args()
    main(args)
