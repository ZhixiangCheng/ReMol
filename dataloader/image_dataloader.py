from numpy import load
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import numpy as np


def train_test_val_idx(dataroot, dataset, split_name):

    split = load(dataroot + '{}/splits/scaffold-{}.npy'.format(dataset, split_name), allow_pickle=True)

    train_idx = split[0]
    val_idx = split[1]
    test_idx = split[2]

    data = pd.read_csv(dataroot + '{}/{}.csv'.format(dataset, dataset))

    task_names = data.columns.drop(['smiles']).tolist()
    label_values = np.array(data[task_names].values, dtype=np.float32)

    train_smi = data.iloc[train_idx]["smiles"].tolist()
    val_smi = data.iloc[val_idx]["smiles"].tolist()
    test_smi = data.iloc[test_idx]["smiles"].tolist()

    train_label = label_values[train_idx]
    val_label = label_values[val_idx]
    test_label = label_values[test_idx]

    return train_smi, train_label, val_smi, val_label, test_smi, test_label


def sample_n_or_all(x, n, seed):
    if len(x) < n:
        return x
    return x.sample(n, random_state=seed)


def process(dataroot, name, k, runseed):
    data = pd.read_csv(dataroot + '{}_processed.csv'.format(name))

    if name == "train":
        # sample k from train dataset
        data = data.groupby('reaction_type').apply(lambda x: sample_n_or_all(x, k, runseed)).reset_index(drop=True)


    reactant = data["reactant_smiles"].tolist()
    product = data["prod_smiles"].tolist()
    label = data["reaction_type"].tolist()

    return reactant, product, label


def reaction_class_pred(dataroot, k, runseed):
    train_reactant, train_product, train_label = process(dataroot, "train", k, runseed)
    val_reactant, val_product, val_label = process(dataroot, "valid", k, runseed)
    test_reactant, test_product, test_label = process(dataroot, "test", k, runseed)

    return train_reactant, train_product, train_label, val_reactant, val_product, val_label, test_reactant, test_product, test_label


def smiles_label_cliffs(dataroot, dataset):
    data = pd.read_csv(dataroot + '{}/{}.csv'.format(dataset, dataset))

    train_smi = data[data['split'] == "train"]['smiles'].tolist()
    test_smi = data[data['split'] == "test"]['smiles'].tolist()

    train_label = data[data['split'] == "train"]['y'].tolist()
    test_label = data[data['split'] == "test"]['y'].tolist()

    cliff_mols_train = data[data['split'] == 'train']['cliff_mol'].tolist()
    cliff_mols_test = data[data['split'] == 'test']['cliff_mol'].tolist()

    return train_smi, train_label, test_smi, test_label, cliff_mols_train, cliff_mols_test


class ReactionDataset(Dataset):
    def __init__(self, reactants, products, labels, img_transformer=None, normalize=None):
        self.reactants = reactants
        self.products = products
        self.labels = labels
        self.normalize = normalize
        self.img_transformer = img_transformer

    def __len__(self):
        return len(self.labels)

    def get_image(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
        img = Image.frombytes("RGB", img.size, img.tobytes())

        return self.img_transformer(img)


    def __getitem__(self, index):
        reactants = self.reactants[index]
        label = self.labels[index]
        product = self.products[index]
        reaction_li = reactants.split(".")

        reactant_img = [self.get_image(smi) for smi in reaction_li]
        product_img = self.get_image(product)

        if self.normalize is not None:
            reactant_img = [self.normalize(img) for img in reactant_img]
            product_img = self.normalize(product_img)

        return {
            "reactant_img": reactant_img,
            "product_img": product_img,
            "label": label,
        }



def my_collate_fn(batch):
    max_reactants = max(len(item['reactant_img']) for item in batch)

    # 初始化批处理数据结构
    batch_data = {
        "reactant_img": [],
        "product_img": [],
        "label": [],
        "masks": []
    }

    padd_img = torch.randn((3, 224, 224))

    for item in batch:
        num_reactants = len(item["reactant_img"])
        num_padding = max_reactants - num_reactants
        batch_data["reactant_img"].append(item["reactant_img"] + [padd_img] * num_padding)
        batch_data["product_img"].append(item["product_img"])
        batch_data["label"].append(item["label"])
        batch_data["masks"].append([0] * num_reactants + [1] * num_padding)

    batch_data["reactant_img"] = torch.stack(
        [torch.stack(imgs) for imgs in batch_data["reactant_img"]])  # B * max_reactants * C * H * W
    batch_data["product_img"] = torch.stack(batch_data["product_img"])  # B * C * H * W
    batch_data["label"] = torch.tensor(batch_data["label"])  # B
    batch_data["masks"] = torch.stack(
        [torch.tensor(x, dtype=torch.bool) for x in batch_data["masks"]])  # B * max_reactants

    return batch_data



class ImageDataset(Dataset):
    def __init__(self, smiles, labels, img_transformer=None, normalize=None):
        self.smiles = smiles
        self.labels = labels
        self.normalize = normalize
        self.img_transformer = img_transformer

    def __len__(self):
        return len(self.smiles)

    def get_image(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
        img = Image.frombytes("RGB", img.size, img.tobytes())

        return self.img_transformer(img)

    def __getitem__(self, index):
        smi = self.smiles[index]
        label = self.labels[index]
        img = self.get_image(smi)

        if self.normalize is not None:
            img = self.normalize(img)

        return img, label
