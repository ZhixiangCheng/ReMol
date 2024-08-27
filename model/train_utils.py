import os
import random
import numpy as np
import torch
from sklearn import metrics
import logging
import torch.nn as nn
import timm
import torch
import torchvision
from collections import defaultdict
from typing import List, Union


class ReMol(nn.Module):
    def __init__(self, dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=1):
        super(ReMol, self).__init__()
        self.num_tasks = num_tasks
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 1024)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = "cls"
        self.to_latent = nn.Identity()

        self.regressor = nn.Linear(dim, self.num_tasks)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        b, C, H, W = img.shape
        x = self.resnet(img)
        x = x.unsqueeze(1)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1).cuda()
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[0, :, :]   # [batch_size, feature_dim]
        x = self.to_latent(x)


        x = self.leakyReLU(x)
        x = self.regressor(x)

        return x



class ReMol_Reaction(nn.Module):
    def __init__(self, dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=1):
        super(ReMol_Reaction, self).__init__()
        self.num_tasks = num_tasks
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 1024)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = "cls"
        self.to_latent = nn.Identity()

        self.regressor = nn.Linear(dim*2, self.num_tasks)
        self.leakyReLU = nn.LeakyReLU()


    def process(self, img, device, mask):
        b, m, C, H, W = img.shape
        reactant_imgs = img.view(-1, C, H, W)
        features = self.resnet(reactant_imgs)  # [batch_size * max_reactants, feature_dim]
        x = features.view(b, m, -1)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        cls_mask = torch.zeros(b, 1, dtype=torch.bool).to(device)
        padding_mask = torch.cat((cls_mask, mask), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)  # [batch_size, max_reactants+1, feature_dim]
        x = x.permute(1, 0, 2)  # [max_reactants+1, batch_size, feature_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[0, :, :]  # [batch_size, feature_dim]
        x = self.to_latent(x)

        return x


    def forward(self, reactant, product, mask, device):
        b, C, H, W = product.shape
        x = self.resnet(product)
        x = x.unsqueeze(1)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1).cuda()
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[0, :, :]   # [batch_size, feature_dim]

        product_embedding = self.to_latent(x)
        reactant_embedding = self.process(reactant, device, mask)

        reaction_embedding = torch.cat((reactant_embedding, product_embedding), dim=1)

        x = self.leakyReLU(reaction_embedding)
        x = self.regressor(x)

        return x



class ReMol_Product(nn.Module):
    def __init__(self, dim=1024, depth=2, heads=2, emb_dropout=0.1, pretrained=False, num_tasks=1):
        super(ReMol_Product, self).__init__()
        self.num_tasks = num_tasks
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 1024)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = "cls"
        self.to_latent = nn.Identity()

        self.regressor = nn.Linear(dim*2, self.num_tasks)
        self.leakyReLU = nn.LeakyReLU()


    def forward(self, img, mask, device, reactant):

        if reactant:
            b, m, C, H, W = img.shape
            reactant_imgs = img.view(-1, C, H, W)
            features = self.resnet(reactant_imgs)  # [batch_size * max_reactants, feature_dim]
            x = features.view(b, m, -1)

            cls_tokens = self.cls_token.expand(b, -1, -1)
            cls_mask = torch.zeros(b, 1, dtype=torch.bool).to(device)
            padding_mask = torch.cat((cls_mask, mask), dim=1)
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.dropout(x)  # [batch_size, max_reactants+1, feature_dim]
            x = x.permute(1, 0, 2)  # [max_reactants+1, batch_size, feature_dim]
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            b, C, H, W = img.shape
            x = self.resnet(img)
            x = x.unsqueeze(1)
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1).to(device)
            x = self.dropout(x)
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[0, :, :]  # [batch_size, feature_dim]
        x = self.to_latent(x)

        return x



def fix_train_random_seed(seed=2023):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_finetune_ckpt(resume, model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None,
                       result_dict=None,
                       logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler,
        'loss': loss,
        'result_dict': result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))

    filename = '{}/{}_{}.pth'.format(save_path, resume, filename_pre)
    torch.save(state, filename)
    log.info('model has been saved as {}'.format(filename))


def metric(y_true, y_pred, y_prob, empty=-1):
    '''
    for classification evaluation on single task
    :param y_true: 1-D, e.g. [1, 0, 1, 1]
    :param y_pred: 1-D, e.g. [0, 0, 1, 1]
    :param y_prob: 1-D, e.g. [0.7, 0.5, 0.2, 0.7]
    :return:
    '''
    assert len(y_true) == len(y_pred) == len(y_prob)
    y_true, y_pred, y_prob = np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob).flatten()
    # filter empty data
    flag = y_true != empty
    y_true, y_pred, y_prob = y_true[flag], y_pred[flag], y_prob[flag]

    auc = metrics.roc_auc_score(y_true, y_prob)
    return {"ROCAUC": auc}


def metric_multitask(y_true, y_pred, y_prob, num_tasks, empty=-1):
    '''
    :param y_true: ndarray, shape is [batch, num_tasks]
    :param y_pred: ndarray, shape is [batch, num_tasks]
    :param y_prob: ndarray, shape is [batch, num_tasks]
    :return:
    '''
    assert num_tasks == y_true.shape[1] == y_pred.shape[1] == y_prob.shape[1]
    assert y_prob.min() >= 0 and y_prob.max() <= 1

    result_list_dict_each_task = []

    cur_num_tasks = 0
    for i in range(num_tasks):
        flag = y_true[:, i] != empty
        if len(set(y_true[flag, i].flatten())) == 1:  # labels are all one value
            result_list_dict_each_task.append(None)
        else:
            result_list_dict_each_task.append(
                metric(y_true[flag, i].flatten(), y_pred[flag, i].flatten(), y_prob[flag, i].flatten()))
            cur_num_tasks += 1

    mean_performance = defaultdict(float)

    for i in range(num_tasks):
        if result_list_dict_each_task[i] is None:
            continue
        for key in result_list_dict_each_task[i].keys():
            if key == "fpr" or key == "tpr" or key == "precision_list" or key == "recall_list":
                continue
            mean_performance[key] += result_list_dict_each_task[i][key] / cur_num_tasks

    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task

    # if cur_num_tasks < num_tasks:
    #     print("Some target is missing! Missing ratio: {:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
    #                                                                          cur_num_tasks, num_tasks))
    #     mean_performance["some_target_missing"] = "{:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
    #                                                                       cur_num_tasks, num_tasks)

    return mean_performance


def metric_reg(y_true, y_pred):
    '''
    for regression evaluation on single task
    :param y_true: 1-D, e.g. [1.1, 0.2, 1.5, 3.2]
    :param y_pred: 1-D, e.g. [-0.2, 1.1, 1.2, 3.1]
    :return:
    '''
    assert len(y_true) == len(y_pred)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = metrics.r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)