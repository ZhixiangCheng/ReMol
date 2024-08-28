# ReMol
![image](img/ReMol.png)


## Install environment

**1. GPU environmentx**<br>  
CUDA 11.1

**2. create a new conda environment**<br>  
conda create -n ReMol python=3.7<br>  
conda activate ReMol  

**3. download some packages**<br>  
pip install -r requirements.txt<br>  
source activate ReMol  

## Pretraining
Download [pretraining data](https://drive.google.com/file/d/188ulJv3Vz8p75hB2GSrsSFfgYrJMv7KG/view?usp=drive_link) and put it into ./datasets/pretrain/<br>  
**1. get masking atom/bond image for mask reaction center identification**<br>  
```
python ./process/mask_center_lmdb.py.py --jobs 15
```
**Note:** You can find the uspto_pretrain in ./datasets/pretrain, and we provide uspto_pretrain_toy in our pretraining data.<br>  

**2. start to pretrain**<br>  
Code to pretrain:<br>  
```
python pretrain.py --nodes 1 \
                   --ngpus_per_node 4 \
                   --gpu 0,1,2,3 \
                   --batch 128 \
                   --epochs 50 \
                   --RPCL_lambda 1 \
                   --RRCL_lambda 1 \
                   --MRCI_lambda 1 \
```
## Finetuning
|                                  |             |                        |             |
|----------------------------------|-------------|------------------------|-------------|
| Task                             | Dataset     | Original download link | Description |
| reaction product prediction      | USPTO-479k  |                        |             |
| chemical reaction classification | Schneider   |                        |             |
| molecular property prediction    | MoleculeNet |                        |             |
| activity cliff estimation        | MoleculeACE |                        |             |


All processed finetuning datasets can be download in [link](https://drive.google.com/file/d/1I2O0AhTO3CGaYMsQl_EnKusmJw6ZXC8y/view?usp=sharing)
