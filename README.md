# GreenAug: Green Screen Augmentation Enables Scene Generalisation

[Eugene Teoh](https://eugeneteoh.com/), [Sumit Patidar](https://rocketsumit.github.io/), [Xiao Ma](https://yusufma03.github.io/), [Stephen James](https://stepjam.github.io/)

[Website](https://greenaug.github.io/), [Paper](https://arxiv.org/abs/2407.07868)

Code will be released before Monday, 15 July 2024 09:00 PDT. No lies!

## Usage

Install GreenAug as a python package:

```bash
pip install git+https://github.com/eugeneteoh/greenaug.git
```

Example usage:

```python
import torch
from torchvision.transforms import v2
from greenaug import GreenAugRandom

transforms = v2.Compose([
    ...
    GreenAugRandom(...)
    ...
])
```


```bash
conda create -n greenaug python=3.10 -y
conda activate greenaug
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda activate greenaug

# Install Pytorch
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install cuda-toolkit -c nvidia/label/cuda-12.1.1 -y

pip install -e .
```
