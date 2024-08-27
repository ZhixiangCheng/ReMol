# -*- coding: UTF-8 -*-
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import io
from io import BytesIO
import pickle
from tqdm import tqdm
import time
import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing
import lmdb
import warnings

warnings.filterwarnings('ignore')

"""
# reaction
[
    # Mutil-Reactants
    [
       [
         ["reactant img  Byte"], # img
         [
             ("atom_mask_img Byte", "label"),(),...
         ], # atom
         [
             ("bond_mask_img Byte", "label"),(),...
         ], # bond
       ],
       [],
        ...
    ],
    # One Product
    [
        ["product img  Byte"], # img
         [
             ("atom_mask_img Byte", "label"),(),...
         ], # atom
         [
             ("bond_mask_img Byte", "label"),(),...
         ], # bond
    ],
    # label
    label
]

"""


def mask_with_atom_and_bound_index(mol, w, h, atom_list, atom_colour_list, bond_list, bound_colour_list,
                                   radius_list, path="", save_svg=False):
    if not save_svg:
        d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
        d2d.drawOptions().useBWAtomPalette()
        d2d.drawOptions().highlightBondWidthMultiplier = 20
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=atom_list,
                                           highlightAtomColors=atom_colour_list,
                                           highlightBonds=bond_list,
                                           highlightBondColors=bound_colour_list,
                                           highlightAtomRadii=radius_list)
        image_data = d2d.GetDrawingText()

        # Convert the byte array to a PIL Image object
        image = Image.open(io.BytesIO(image_data))

        # Convert the PIL Image object to a NumPy array
        image_np = np.array(image)

        return image_np


    else:
        d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=atom_list,
                                           highlightAtomColors=atom_colour_list,
                                           highlightBonds=bond_list,
                                           highlightBondColors=bound_colour_list,
                                           highlightAtomRadii=radius_list)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        with open(path, 'w') as f:
            f.write(svg)


def mask_atom(img):
    ball_color = 'green'
    color_dist = {
        'green': {'Lower': np.array([30, 66, 35]), 'Upper': np.array([85, 255, 255])},
    }

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img1, (7, 7), 0)
    inRange_hsv = cv2.inRange(blurred, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])

    return inRange_hsv


def mask_bond(img):
    ball_color = 'green'
    color_dist = {
        'green': {'Lower': np.array([30, 120, 40]), 'Upper': np.array([60, 255, 255])},
    }

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(img1, (7, 7), 0)

    inRange_hsv = cv2.inRange(blurred, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 定义结构元素的形状和大小
    dilated = cv2.dilate(inRange_hsv, kernel)

    return inRange_hsv


def img2byte(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_byte = buffer.getvalue()
    return img_byte


def get_img(mol, w=224, h=224):
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol)
    image_data = d2d.GetDrawingText()

    # Convert the byte array to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Convert the PIL Image object to a NumPy array
    image_np = np.array(image)

    img_byte = img2byte(image)

    return image_np, img_byte


def get_color_dict(atom, radius=0.4, color=(0, 1, 0)):
    radius_dict = {}
    atom_colour_dict = {}
    atom_index = []

    # radiux
    radius_dict[atom] = radius

    # color
    atom_colour_dict[atom] = color

    # atom/bond index
    atom_index.append(atom)

    return atom_colour_dict, radius_dict, atom_index


def get_atom_img(mol, atom):
    img, _ = get_img(mol, w=224, h=224)
    atom_colour_dict, radius_dict, atom_index = get_color_dict(atom, radius=0.4, color=(0, 1, 0))
    highlight_img = mask_with_atom_and_bound_index(mol, w=224, h=224,
                                                   atom_list=atom_index,
                                                   atom_colour_list=atom_colour_dict,
                                                   bond_list=None,
                                                   bound_colour_list=None,
                                                   radius_list=radius_dict,
                                                   save_svg=False)

    dilated_atom = mask_atom(highlight_img)

    atom_mask = cv2.cvtColor(dilated_atom, cv2.COLOR_RGB2BGR)

    # img[:, :, :][atom_mask[:, :, :] > 0] = 255

    atom_img = Image.fromarray(atom_mask)
    img_byte = img2byte(atom_img)

    return img_byte


def get_bond_img(mol, bond):
    img, _ = get_img(mol, w=224, h=224)
    bond_colour_dict, radius_dict, bond_index = get_color_dict(bond, radius=0.4, color=(0, 1, 0))
    highlight_img = mask_with_atom_and_bound_index(mol, w=224, h=224,
                                                   atom_list=None,
                                                   atom_colour_list=None,
                                                   bond_list=bond_index,
                                                   bound_colour_list=bond_colour_dict,
                                                   radius_list=radius_dict,
                                                   save_svg=False)

    dilated_bond = mask_bond(highlight_img)

    bond_mask = cv2.cvtColor(dilated_bond, cv2.COLOR_RGB2BGR)

    # img[:, :, :][bond_mask[:, :, :] > 0] = 255

    bond_img = Image.fromarray(bond_mask)
    img_byte = img2byte(bond_img)

    return img_byte


def mask_atom_bond(smiles, hit_atom, hit_bond):
    mol = Chem.MolFromSmiles(smiles)
    _, molecule = get_img(mol, w=224, h=224)

    Atom = [atom.GetIdx() for atom in mol.GetAtoms()]
    Bond = [bond.GetIdx() for bond in mol.GetBonds()]

    # define atom and bond label, if atom/bond is center, label is 1, or  0
    atom_label = [1 if atom in hit_atom else 0 for atom in Atom]
    # mask atom center
    atom_img = [get_atom_img(mol, atom) for atom in Atom]

    # atom_img_label = list(zip(atom_img, atom_label))

    # product maybe not have bond center
    if len(Bond) == 0 and len(hit_bond) == 0:
        bond_label = [0]
        _, img = get_img(mol, w=224, h=224)
        bond_img = [img]

    elif len(Bond) != 0 and len(hit_bond) == 0:
        bond_img = [get_bond_img(mol, bond) for bond in Bond]
        bond_label = [0] * len(Bond)

    else:
        bond_label = [1 if bond in hit_bond else 0 for bond in Bond]

        # mask bond center
        bond_img = [get_bond_img(mol, bond) for bond in Bond]

    # bond_img_label = list(zip(bond_img, bond_label))

    return molecule, atom_img, atom_label, bond_img, bond_label


def process_reactant_product(rxn):
    smiles, hit_atom, hit_bond = rxn.split("//")

    hit_atom = list(map(lambda x: int(x), hit_atom.split("_")))
    # bond center exist
    hit_bond = list(map(lambda x: int(x), hit_bond.split("_"))) if hit_bond else hit_bond

    return smiles, hit_atom, hit_bond


def process_reaction(index, reactant_li, product_li, label):
    value_dict = {
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
    }

    reactant = reactant_li[index]
    for reactant in reactant.split("."):
        reactant_smiles, reactant_hit_atom, reactant_hit_bond = process_reactant_product(reactant)
        reactant_img, reactant_mask_atom, reactant_mask_atom_label, reactant_mask_bond, reactant_mask_bond_label = mask_atom_bond(
            reactant_smiles, reactant_hit_atom, reactant_hit_bond)
        value_dict["reactant_img"].append(reactant_img)
        value_dict["reactant_mask_atom"].append(reactant_mask_atom)
        value_dict["reactant_mask_atom_label"].append(reactant_mask_atom_label)
        value_dict["reactant_mask_bond"].append(reactant_mask_bond)
        value_dict["reactant_mask_bond_label"].append(reactant_mask_bond_label)

    product_smiles, product_hit_atom, product_hit_bond = process_reactant_product(product_li[index])
    product_img, product_mask_atom, product_mask_atom_label, product_mask_bond, product_mask_bond_label = mask_atom_bond(
        product_smiles, product_hit_atom, product_hit_bond)
    value_dict["product_img"].append(product_img)
    value_dict["product_mask_atom"].extend(product_mask_atom)
    value_dict["product_mask_atom_label"].extend(product_mask_atom_label)
    value_dict["product_mask_bond"].extend(product_mask_bond)
    value_dict["product_mask_bond_label"].extend(product_mask_bond_label)

    value_dict["label"] = label[index]

    return (index, value_dict)


def write_lmdb(results, txn):
    for index, value_dict in results:
        # Construct the key-value pair
        key = str(index).encode()
        value = pickle.dumps(value_dict)
        # Store the key-value pair in the database
        txn.put(key, value)
    txn.commit()

def main(args):
    csv_path = "/data/chengzhixiang/ReMol/uspto_pretrain.csv"
    # csv_path = "./USPTO/processed/uspto_pretrain.csv"

    data = pd.read_csv(csv_path)

    start = args.start
    end = args.end

    reactant_li = data["reactant_hit_atom_bond"].tolist()[start:end]
    product_li = data["product_hit_atom_bond"].tolist()[start:end]
    label = data["label"].tolist()[start:end]

    # # test
    # reactant_li = ["Oc1ccc(Cl)cc1//0_1//0.ClCCBr//3_2_1//2_1"]
    # product_li = ["[Cl-]//0//"]
    # label = [0]

    # parallel
    num_cores = args.jobs  # 根据您的系统情况设置核心数
    results = Parallel(n_jobs=num_cores)(
        delayed(process_reaction)(i, reactant_li, product_li, label) for i in tqdm(range(len(reactant_li))))

    env_reaction = lmdb.open(
        "/data/chengzhixiang/ReMol/uspto_pretrain_lmdb_{}".format(end),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
        map_size=1099511627776)
    txn = env_reaction.begin(write=True)
    write_lmdb(results, txn)
    env_reaction.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters of Mask Data Process')

    parser.add_argument('--start', default=0, type=int, help='strat')
    parser.add_argument('--end', default=100000, type=int, help='end')
    parser.add_argument('--jobs', default=10, type=int, help='n_jobs')

    args = parser.parse_args()
    main(args)
