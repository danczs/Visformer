# Visformer
![pytorch](https://img.shields.io/badge/pytorch-v1.7.0-green.svg?style=plastic)

## Introduction
This is a pytorch implementation for the Visformer models. This project is based on the training code in [Deit](https://github.com/facebookresearch/deit) and the tools in [timm](https://github.com/rwightman/pytorch-image-models).

## Usage
Clone the repository:
```bash
git clone https://github.com/danczs/Visformer.git
```
Install pytorch, timm and einops:
```bash
pip install -r requirements.txt
```
## Data preparation
The layout of Imagenet data:
```bash
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
```
## Network Training
Visformer_small
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model visformer_small --batch-size 64 --data-path /path/to/imagenet --output_dir /path/to/save
```
Visformer_tiny
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model visformer_tiny --batch-size 256 --drop-path 0.0 --data-path /path/to/imagenet --output_dir /path/to/save
```
For the current version, visformer_small can achieve 82.28% on ImageNet. 
