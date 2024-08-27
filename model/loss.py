import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch.distributed.nn
from torch import distributed as dist


def gather_features(image_features, world_size, rank=0, local_loss=False):
    # https://github.com/mlfoundations/open_clip/blob/3ef1e923200fc237717e3ff31998903e919f0c4a/src/open_clip/loss.py
    with torch.no_grad():
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
    if not local_loss:
        # ensure grads for local rank when all_* features don't have a gradient
        gathered_image_features[rank] = image_features
    all_image_features = torch.cat(gathered_image_features, dim=0)

    return all_image_features


def calculate_loss(reactant_embeddings, product_embeddings, temperature=0.05):
    """
    Calculate the contrastive loss using InfoNCE.

    :param reactant_embeddings: Embeddings for reactants.
    :param product_embeddings: Embeddings for products.
    :param temperature: Temperature parameter for scaling the logits.
    :return: Contrastive loss value.
    """

    # calculate distance
    distances = torch.cdist(reactant_embeddings, product_embeddings, p=2)

    # loss
    logits = -distances / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def getTanimotocoefficient(s, t):
    s = np.asarray(s)
    t = np.asarray(t)
    if (s.shape != t.shape):
        print("Shape unvalid")
        return -1
    return (np.sum(s * t)) / (np.sum(s ** 2) + np.sum(t ** 2) - np.sum(s * t))


def get_tom_matrix(X):
    # 初始化一个零矩阵来存储所有样本之间的相似度
    similarity_matrix = np.zeros((len(X), len(X)))

    # 计算相似度矩阵
    for i in range(len(X)):
        for j in range(len(X)):
            if i <= j:  # 谷本相似度是对称的，所以只需要计算一半然后复制到另一半
                similarity = getTanimotocoefficient(X[i], X[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return similarity_matrix


def mse_loss(reaction_embeddings, label, t=0.05):
    similarity_matrix = F.cosine_similarity(reaction_embeddings.unsqueeze(1), reaction_embeddings.unsqueeze(0),
                                            dim=2).to(label.device)

    label2 = torch.tensor(get_tom_matrix(label.cpu().numpy()), dtype=torch.float32).to(label.device)

    similarity_matrix /= t
    label2 /= t

    criterion2 = nn.MSELoss()

    loss = criterion2(similarity_matrix.to(torch.float32), label2.to(torch.float32)) / 2

    return loss.float()
