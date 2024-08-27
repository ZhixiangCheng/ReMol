import random
import torch
import cv2
import lmdb
import pickle
import itertools
from PIL import Image
import numpy as np
from io import BytesIO
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MoleculeDataset(Dataset):
    def __init__(self, index, data_path, transform=None):
        self.index = index
        self.data_path = data_path
        self.transform = transform
        self.env = None
        self.txn_reaction = None

    def __len__(self):
        return len(self.index)

    def _init_db(self):
        self.env_reaction = lmdb.open(self.data_path, subdir=False, readonly=True,
                                      lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn_reaction = self.env_reaction.begin()
        self.lmdb_env_reaction = {'txn': self.txn_reaction,
                                  'keys': list(self.txn_reaction.cursor().iternext(values=False))}

    def PIL2Numpy(self, pil_image):
        # Convert PIL image to numpy array
        numpy_image = np.array(pil_image)
        # Convert RGB to BGR
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image

    def mask(self, data, image):

        img_mask = self.PIL2Numpy(self.Byte2PIL(data[0][0]))
        image[:, :, :][img_mask[:, :, :] > 0] = 255
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return image, data[0][1]

    def get_mask(self, img, label, image):
        image = self.PIL2Numpy(self.Byte2PIL(image))
        data = list(zip(img, label))

        data_0 = [item for item in data if item[1] == 0]
        data_1 = [item for item in data if item[1] == 1]

        # chose 2 images from data
        sampled_data_0 = random.sample(data_0, 1) if data_0 else []
        sampled_data_1 = random.sample(data_1, 1) if data_1 else []

        if sampled_data_0 and sampled_data_1:
            # all are center
            image0, data0 = self.mask(sampled_data_0, image)
            image1, data1 = self.mask(sampled_data_1, image)
            return [image0, image1], [data0, data1]
        elif sampled_data_1:
            # only center
            image1, data1 = self.mask(sampled_data_1, image)
            return [image1], [data1]
        else:
            image0, data0 = self.mask(sampled_data_0, image)
            return [image0], [data0]

    def Byte2PIL(self, img):
        return Image.open(BytesIO(img))

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        k = self.index[idx]

        reaction_datapoint_pickled = self.txn_reaction.get(str(k).encode())
        reaction = pickle.loads(reaction_datapoint_pickled)

        reactant_img = reaction["reactant_img"]
        product_img = reaction["product_img"][0]
        label = reaction["label"]

        reactant_mask_atom, reactant_mask_atom_label = zip(*[
            self.get_mask(atom, label, image)
            for atom, label, image in
            zip(reaction["reactant_mask_atom"], reaction["reactant_mask_atom_label"], reactant_img)
        ])
        reactant_mask_atom = list(itertools.chain(*reactant_mask_atom))
        reactant_mask_atom_label = list(itertools.chain(*reactant_mask_atom_label))

        reactant_mask_bond, reactant_mask_bond_label = zip(*[
            self.get_mask(bond, label, image)
            for bond, label, image in
            zip(reaction["reactant_mask_bond"], reaction["reactant_mask_bond_label"], reactant_img)
        ])

        reactant_mask_bond = list(itertools.chain(*reactant_mask_bond))
        reactant_mask_bond_label = list(itertools.chain(*reactant_mask_bond_label))

        product_mask_atom, product_mask_atom_label = self.get_mask(reaction["product_mask_atom"],
                                                                   reaction["product_mask_atom_label"], product_img)
        product_mask_bond, product_mask_bond_label = self.get_mask(reaction["product_mask_bond"],
                                                                   reaction["product_mask_bond_label"], product_img)

        # image = self.Byte2PIL(product_img)
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()

        if self.transform:
            reactant_img = [self.transform(self.Byte2PIL(img)) for img in reactant_img]
            product_img = self.transform(self.Byte2PIL(product_img))

            reactant_mask_atom = [self.transform(tpl) for tpl in reactant_mask_atom]
            reactant_mask_bond = [self.transform(tpl) for tpl in reactant_mask_bond]
            product_mask_atom = [self.transform(tpl) for tpl in product_mask_atom]
            product_mask_bond = [self.transform(tpl) for tpl in product_mask_bond]

        return {
            "reactant_img": reactant_img,
            "product_img": product_img,
            "label": label,
            "reactant_mask_atom": reactant_mask_atom,
            "reactant_mask_atom_label": reactant_mask_atom_label,
            "reactant_mask_bond": reactant_mask_bond,
            "reactant_mask_bond_label": reactant_mask_bond_label,
            "product_mask_atom": product_mask_atom,
            "product_mask_atom_label": product_mask_atom_label,
            "product_mask_bond": product_mask_bond,
            "product_mask_bond_label": product_mask_bond_label,
            "reaction_fp": reaction["reactopn_fp"],
        }


def getTanimotocoefficient(s,t):
    s=np.asarray(s)
    t=np.asarray(t)
    if (s.shape!=t.shape):
        print("Shape unvalid")
        return -1
    return (np.sum(s*t))/(np.sum(s**2)+np.sum(t**2)-np.sum(s*t))


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

def my_collate_fn(batch):
    max_reactants = max(len(item['reactant_img']) for item in batch)

    # 初始化批处理数据结构
    batch_data = {
        "reactant_img": [],
        "product_img": [],
        "label": [],
        "reactant_mask_atom": [],
        "reactant_mask_atom_label": [],
        "reactant_mask_bond": [],
        "reactant_mask_bond_label": [],
        "product_mask_atom": [],
        "product_mask_atom_label": [],
        "product_mask_bond": [],
        "product_mask_bond_label": [],
        "masks": [],
        "reaction_fp_sim": []
    }

    padd_img = torch.randn((3, 224, 224))

    smi_li= []

    for item in batch:
        num_reactants = len(item["reactant_img"])
        num_padding = max_reactants - num_reactants

        batch_data["reactant_img"].append(item["reactant_img"] + [padd_img] * num_padding)
        batch_data["reactant_mask_atom"].extend([tpl for tpl in item["reactant_mask_atom"]])
        batch_data["reactant_mask_atom_label"].extend([tpl for tpl in item["reactant_mask_atom_label"]])
        batch_data["reactant_mask_bond"].extend([tpl for tpl in item["reactant_mask_bond"]])
        batch_data["reactant_mask_bond_label"].extend([tpl for tpl in item["reactant_mask_bond_label"]])

        batch_data["product_img"].append(item["product_img"])
        # batch_data["product_mask_atom"].append(item["product_mask_atom"])
        # batch_data["product_mask_atom_label"].append(item["product_mask_atom_label"])
        # batch_data["product_mask_bond"].append(item["product_mask_bond"])
        # batch_data["product_mask_bond_label"].append(item["product_mask_bond_label"])
        batch_data["product_mask_atom"].extend([tpl for tpl in item["product_mask_atom"]])
        batch_data["product_mask_atom_label"].extend([tpl for tpl in item["product_mask_atom_label"]])
        batch_data["product_mask_bond"].extend([tpl for tpl in item["product_mask_bond"]])
        batch_data["product_mask_bond_label"].extend([tpl for tpl in item["product_mask_bond_label"]])

        batch_data["label"].append(item["label"])
        # Transformer src_key_padding_mask is True, padding
        batch_data["masks"].append([0] * num_reactants + [1] * num_padding)

        smi_li.append(item["reaction_fp"])

    similarity_matrix = get_tom_matrix(smi_li)

    batch_data["reactant_img"] = torch.stack(
        [torch.stack(imgs) for imgs in batch_data["reactant_img"]])  # B * max_reactants * C * H * W
    batch_data["product_img"] = torch.stack(batch_data["product_img"])  # B * C * H * W
    batch_data["label"] = torch.tensor(batch_data["label"])  # B
    batch_data["masks"] = torch.stack(
        [torch.tensor(x, dtype=torch.bool) for x in batch_data["masks"]])  # B * max_reactants

    batch_data["reactant_mask_atom"] = torch.stack(
        batch_data["reactant_mask_atom"])  # total_reactants_batch * C * H * W
    batch_data["reactant_mask_atom_label"] = torch.tensor(batch_data["reactant_mask_atom_label"])
    batch_data["reactant_mask_bond"] = torch.stack(batch_data["reactant_mask_bond"])
    batch_data["reactant_mask_bond_label"] = torch.tensor(batch_data["reactant_mask_bond_label"])

    batch_data["product_mask_atom"] = torch.stack(batch_data["product_mask_atom"])  # B * C * H * W
    batch_data["product_mask_atom_label"] = torch.tensor(batch_data["product_mask_atom_label"])
    batch_data["product_mask_bond"] = torch.stack(batch_data["product_mask_bond"])
    batch_data["product_mask_bond_label"] = torch.tensor(batch_data["product_mask_bond_label"])

    batch_data["reaction_fp_sim"] = torch.tensor(similarity_matrix)

    return batch_data


if __name__ == '__main__':
    import torch
    import numpy as np
    from tqdm import tqdm
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split

    nums = 100
    data_path = "../USPTO/processed/uspto_pretrain_lmdb_toy"

    transform = [transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                 transforms.ToTensor()]

    train_index, val_index = train_test_split(list(range(nums)), test_size=0.2, shuffle=True, random_state=2023)

    train_dataset = MoleculeDataset(train_index, data_path, transforms.Compose(transform))
    val_dataset = MoleculeDataset(val_index, data_path, transforms.Compose(transform))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                                   num_workers=1, pin_memory=True, collate_fn=my_collate_fn)

    with tqdm(total=len(train_dataloader), position=0, ncols=120) as t:
        for i, tmp in enumerate(train_dataloader):
            print(tmp)
            break
