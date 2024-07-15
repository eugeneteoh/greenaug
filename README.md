# GreenAug: Green Screen Augmentation Enables Scene Generalisation

[Eugene Teoh](https://eugeneteoh.com/), [Sumit Patidar](https://rocketsumit.github.io/), [Xiao Ma](https://yusufma03.github.io/), [Stephen James](https://stepjam.github.io/)

[Website](https://greenaug.github.io/), [Paper](https://arxiv.org/abs/2407.07868)

This repo contains the following augmentation methods:

- `greenaug.greenaug_random.GreenAugRandom`: This applies random textures to the chroma-keyed background. In our paper, we used [mil_data](https://huggingface.co/datasets/eugeneteoh/mil_data).

- `greenaug.greenaug_generative.GreenAugGenerative`: This uses the chroma-keyed mask to inpaint realistic or imagined backgrounds using Stable Diffusion.

- `greenaug.greenaug_mask.GreenAugMask`: This uses a masking network to isolate backgrounds as dark pixels during inference. One first needs to train a masking network (see instructions below).

- `greenaug.generative_augmentation.GenerativeAugmentation`: This is an implementation of generative augmentation (e.g. [CACTI](https://arxiv.org/abs/2212.05711), [GenAug](https://arxiv.org/abs/2302.06671), [ROSIE](https://arxiv.org/abs/2302.11550)). The implementation is close to ROSIE, but with open source models (Grounding DINO, Segment Anything, Stable Diffusion).

These augmentation methods can be integrated during policy learning (imitation or reinforcement). In our experiments, we used [ACT](https://github.com/tonyzhaozh/act) and (Coarse-to-fine Q-Network)[https://github.com/younggyoseo/CQN].

## Installation

Install GreenAug as a Python package:

```bash
pip install greenaug @ git+https://github.com/eugeneteoh/greenaug.git
```

To use the generative variants (GreenAugGenerative and GenerativeAugmentation), set the `CUDA_HOME` environment variable and install `cuda-toolkit`:

```bash
conda create -n greenaug python=3.10 -y
conda activate greenaug
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda activate greenaug

# Install PyTorch
# Follow instructions at https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install cuda-toolkit -c nvidia/label/cuda-12.1.1 -y

pip install greenaug[generative] @ git+https://github.com/eugeneteoh/greenaug.git
```

To use GreenAugMask:

```bash
pip install greenaug[mask] @ git+https://github.com/eugeneteoh/greenaug.git
```

Then see the example below.

## Example Usage

Check examples under [examples/](examples/).

```python
import torch
from greenaug import GreenAugRandom

augmenter = GreenAugRandom()  # This is a torch.nn.Module
out = augmenter(image, ...)
```

Training GreenAugMask masking network:

```bash
# Download data
huggingface-cli download --repo-type dataset eugeneteoh/greenaug --include "GreenScreenDemoCollection/open_drawer_green_screen.mp4" --local-dir "assets/mask/raw/"               
huggingface-cli download --repo-type dataset eugeneteoh/mil_data --include "*.png" --local-dir "assets/mask/background/"               

# Preprocess data
python scripts/preprocess_masking_data.py  

# Train Masking Network
python scripts/train_masking_network.py

# Run example
python examples/greenaug_mask.py --checkpoint /path/to/checkpoint
```
