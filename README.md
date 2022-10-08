# Visformer
![pytorch](https://img.shields.io/badge/pytorch-v1.7.0-green.svg?style=plastic)

## Introduction
This is a pytorch implementation for the Visformer models. This project is based on the training code in [DeiT](https://github.com/facebookresearch/deit) and the tools in [timm](https://github.com/rwightman/pytorch-image-models).

## Usage
Clone the repository:
```bash
git clone https://github.com/danczs/Visformer.git
```
Install pytorch, timm and einops:
```bash
pip install -r requirements.txt
```
## Data Preparation
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
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model visformer_tiny --batch-size 256 --drop-path 0.03 --data-path /path/to/imagenet --output_dir /path/to/save
```

Viformer V2 models
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model swin_visformer_small_v2 --batch-size 64 --data-path /path/to/imagenet --output_dir /path/to/save --amp --qk-scale-factor=-0.5
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model swin_visformer_tiny_v2 --batch-size 256 --drop-path 0.03 --data-path /path/to/imagenet --output_dir /path/to/save --amp --qk-scale-factor=-0.5
```
The model performance:

|        model        | top-1 (%) | FLOPs (G) | paramters (M) | 
|:-------------------:|:---------:|:---------:|:-------------:|
|   Visformer_tiny    |   78.6    |    1.3    |     10.3      |
|  Visformer_tiny_V2  |   79.6    |    1.3    |      9.4      |
|   Visformer_small   |   82.2    |   40.2    |      4.9      |
| Visformer_small_V2  |   83.0    |   23.6    |      4.3      |
| Visformer_medium_V2 |   83.6    |   44.5    |      8.5      |

pre-trained models:

|                       model                       |   model    |   log   | top-1 (%) | 
|:-------------------------------------------------:|:----------:|:-------:|:---------:|
|            Visformer_small (original)             | [github]() | [log]() |   82.21   |
|  Visformer_small  (+ Swin for downstream tasks)   | [github]() | [log]() |   82.34   |
| Visformer_small_v2 (+ Swin for downstream tasks)  | [github]() | [log]() |   83.00   |
| Visformer_medium_v2 (+ Swin for downstream tasks) | [github]() | [log]() |   83.62   |

(In some logs, the model is only tested for the last 50 epochs to save the training time.)

[More information about Visformer V2](https://arxiv.org/abs/2104.12533).

## Object Detection on COCO
The standard self-attention is not efficient for high-reolution inputs, 
so we simply replace the standard self-attention with Swin-attention for object detection. Therefore, Swin Transformer is our directly baseline. 
### Mask R-CNN
| Backbone | sched | box mAP | mask mAP | params | FLOPs | FPS |
| :---: | :---: |  :---: | :---: |  :---: |  :---: | :---: | 
| Swin-T |1x| 42.6 | 39.3 | 48 | 267 | 14.8 |
| Visformer-S | 1x| 43.0 | 39.6 | 60 | 275 | 13.1|
| VisformerV2-S | 1x| 44.8 | 40.7 | 43 | 262 | 15.2 |
|Swin-T |3x + MS|  46.0 | 41.6 | 48 | 367 | 14.8 |
| VisformerV2-S | 3x + MS| 47.8 | 42.5 | 43 | 262 | 15.2 |

### Cascade Mask R-CNN
| Backbone | sched | box mAP | mask mAP | params | FLOPs | FPS |
| :---: | :---: |  :---: | :---: |  :---: |  :---: | :---: |
| Swin-T |1x + MS|  48.1 | 41.7 | 86 | 745 | 9.5 |
| VisformerV2-S |1x + MS|  49.3 | 42.3 | 81 | 740 | 9.6 |
| Swin-T |3x + MS|  50.5 | 43.7 | 86 | 745 | 9.5 |
| VisformerV2-S |3x + MS|  51.6 | 44.1 | 81 | 740 | 9.6 |

This repo only contains the key files for object detection ('./ObjectDetction'). [Swin-Visformer-Object-Detection](https://github.com/danczs/Swin-Visformer-Object-Detection)  is the full detection project.

## Pre-trained Model
Beacause of the policy of our institution, we cannot send the pre-trained models out directly. Thankfully, @[hzhang57](https://github.com/hzhang57)  and @[developer0hye](https://github.com/developer0hye) provides [Visformer_small](https://drive.google.com/drive/folders/18GpH1SeVOsq3_2QGTA5Z_3O1UFtKugEu?usp=sharing) and [Visformer_tiny](https://drive.google.com/file/d/1LLBGbj7-ok1fDvvMCab-Fn5T3cjTzOKB/view?usp=sharing) models trained by themselves.

## Automatic Mixed Precision (amp)
In the original version of Visformer, amp can cause NaN values. We find that the overflow comes from the attention mask:
```python
scale = head_dim ** -0.5
attn = ( q  @ k.transpose(-2,-1) ) * scale
``` 
To avoid overflow, we pre-normalize q & k, and, thus, overall normalize 'attn' with 'head_dim' instead of  'head_dim ** 0.5':
```python
scale = head_dim ** -0.5
attn =  (q * scale) @ (k.transpose(-2,-1) * scale) 
```
Amp training:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model visformer_small --batch-size 64 --data-path /path/to/imagenet --output_dir /path/to/save --amp --qk-scale-factor=-0.5
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model visformer_tiny --batch-size 256 --drop-path 0.03 --data-path /path/to/imagenet --output_dir /path/to/save --amp --qk-scale-factor=-0.5
```
This change won't degrade the training performance. 

Using amp for the original pre-trained models:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model visformer_small --batch-size 64 --data-path /path/to/imagenet --output_dir /path/to/save --eval --resume /path/to/weights --amp
```

## Citing
```bash
@inproceedings{chen2021visformer,
  title={Visformer: The vision-friendly transformer},
  author={Chen, Zhengsu and Xie, Lingxi and Niu, Jianwei and Liu, Xuefeng and Wei, Longhui and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={589--598},
  year={2021}
}
```