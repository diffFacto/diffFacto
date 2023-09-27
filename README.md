# DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion
**[DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion]([https://scade-spacecarving-nerfs.github.io](https://difffacto.github.io))** 

George Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shimin Hu, Ke Li, Leonidas Guibas

ICCV 2023

![Alt text](assets/combined.gif)

## Introduction
We introduce DiffFacto, a novel probabilistic generative model that learns the distribution of shapes with part-level control. We propose a factorization that models independent part style and part configuration distributions, and present a novel cross diffusion network that enables us to generate coherent and plausible shapes under our proposed factorization. Experiments show that our method is able to generate novel shapes with multiple axes of control. It generates plausible and coherent shape, while enabling various downstream editing applications such as shape interpolation, mixing and transformation editing. 

```
@inproceedings{nakayama2023difffacto,
      title={DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion}, 
      author={Kiyohiro Nakayama and Mikaela Angelina Uy and Jiahui Huang and Shi-Min Hu and Ke Li and Leonidas Guibas},
      year={2023},
      booktitle = {International Conference on Computer Vision (ICCV)},
}
```
## Pretrained Models
DiffFacto pretrained models can be downloaded [here](http://download.cs.stanford.edu/orion/scade/pretrained_models.zip).

## Code

### Environment Set-up
```bash
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
pip install python/anchor_diff/metrics/chamfer_dist python/anchor_diff/metrics/emd
```
### Demo
TODO

### Training
TODO
```
python ...
```

## License
This repository is released under MIT License (see LICENSE file for details).
