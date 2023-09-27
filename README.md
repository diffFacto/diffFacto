# DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion
**[DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion]([https://scade-spacecarving-nerfs.github.io](https://difffacto.github.io))** 

George Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shimin Hu, Ke Li, Leonidas Guibas

ICCV 2023

![Alt text](assets/combined.gif)

## Introduction
We introduce DiffFacto, a novel probabilistic generative model that learns the distribution of shapes with part-level control. We propose a factorization that models independent part style and part configuration distributions, and present a novel cross diffusion network that enables us to generate coherent and plausible shapes under our proposed factorization. Experiments show that our method is able to generate novel shapes with multiple axes of control. It generates plausible and coherent shape, while enabling various downstream editing applications such as shape interpolation, mixing and transformation editing. 


## Pretrained Models
DiffFacto pretrained models can be downloaded [here](http://download.cs.stanford.edu/orion/DiffFacto/weights.zip).
## Data
The preprocessed data can be downloaded [here](http://download.cs.stanford.edu/orion/DiffFacto/data.zip).
## Code

### Environment Set-up
```bash
# first clone the repo by 
git clone https://github.com/diffFacto/diffFacto.git && cd diffFacto
# Create conda environment with python 3.8
conda create -n diffFacto python=3.8
conda activate diffFacto
# we use torch 1.12.1 and CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# we insall other dependencies
pip install torchdiffeq==0.0.1 tqdm tensorboardX pypaml plyfile einops numpy==1.23.5 scipy scikit-learn einops
# we install pointnet2_ops 
pip install pointnet2_ops_lib/
# we install chamfer distance and emd metric for evaluation 
pip install python/diffFacto/metrics/chamfer_dist python/diffFacto/metrics/emd
# Lastly, we install diffFacto
pip install -e .
```
Memory efficient attention using [xformer](https://github.com/facebookresearch/xformers) is supported if GPU memory is an issue. Please install a compatible version of xformer to save GPU memory. we have test with version `0.0.15`. 
### Demo
To generate shapes using DiffFacto, please place the pretrained weights under `pretrained/` folder and the `root` flag under `dataset` in each of the config files `configs/gen_[chair/airplane/car/lamp].py` should point to the data directory. Then one can generate shapes by running 
```
python tools/run_net.py --config-file configs/gen_[chair/airplane/car/lamp].py  --task val --prefix [chair/airplane/car/lamp]
```
### Training
We provide the training script for chair category. Other categories' training will come soon! 

```
# Change the root flag to point to the data directory 
# To train the part stylizer and cross diffusion network, please run
python tools/run_net.py --config-file configs/train_chair_stage1.py  --task train --prefix chair_stage1
```
After the first stage training is complete, open the config file `configs/train_chair_stage2.py` and replace the `resume_path` path to the pretrained first stage weight. Then the second stage can be trained by running 
```
python tools/run_net.py --config-file configs/train_chair_stage2.py  --task train --prefix chair_stage2
```
By default the second stage training will run for 4000 epoches, but one can pick a suitable checkpoint in between. 


## License
This repository is released under MIT License (see LICENSE file for details).

## Citation
```
@inproceedings{nakayama2023difffacto,
      title={DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion}, 
      author={Kiyohiro Nakayama and Mikaela Angelina Uy and Jiahui Huang and Shi-Min Hu and Ke Li and Leonidas Guibas},
      year={2023},
      booktitle = {International Conference on Computer Vision (ICCV)},
}
```