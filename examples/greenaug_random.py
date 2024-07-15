import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from torchvision.transforms.v2.functional import resize

from greenaug.greenaug_random import GreenAugRandom

seed = 42


class CustomDataset(Dataset):
    def __init__(self, images, background_root):
        self.images = images
        self.background_paths = sorted(Path(background_root).glob("**/*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        bg_path = random.choice(self.background_paths)
        bg = Image.open(bg_path).convert("RGB")
        bg = np.array(bg)
        return image, bg


video_path = hf_hub_download(
    repo_id="eugeneteoh/greenaug",
    repo_type="dataset",
    filename="GreenScreenDemoCollection/open_drawer_green_screen.mp4",
)

background_root = snapshot_download(
    repo_id="eugeneteoh/mil_data", repo_type="dataset", allow_patterns="*.png"
)

frames, _, _ = read_video(video_path, end_pts=5, pts_unit="sec")

dataset = CustomDataset(frames, background_root)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

augmenter = GreenAugRandom()

images_aug_seq = []
for batch in dataloader:
    images, background = batch
    b, h, w, c = images.shape
    # convert to float32 and scale to [0, 1]
    images = images.float() / 255
    images = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

    background = background.float() / 255
    background = background.permute(0, 3, 1, 2)
    background = resize(background, (h, w), antialias=True)

    images_aug = augmenter(
        images, keycolor=["#439f82"] * b, tola=30, tolb=35, background_image=background
    )
    images_aug_seq.append(images_aug)

images_aug_seq = torch.cat(images_aug_seq, dim=0)
images_aug_seq = (images_aug_seq.permute(0, 2, 3, 1) * 255).to(torch.uint8)

out_path = Path("assets/greenaug_random.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)
write_video(out_path, images_aug_seq, 10)
