import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_video, write_video

from greenaug.greenaug_mask import GreenAugMask

seed = 42

argparse = argparse.ArgumentParser()
argparse.add_argument("--checkpoint", type=str)
args = argparse.parse_args()

video_path = hf_hub_download(
    repo_id="eugeneteoh/greenaug",
    repo_type="dataset",
    filename="GreenScreenDemoCollection/open_drawer_green_screen.mp4",
)

frames, _, _ = read_video(video_path, end_pts=5, pts_unit="sec")

dataset = TensorDataset(frames)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

augmenter = GreenAugMask(checkpoint=args.checkpoint).to("cpu")

images_aug_seq = []
for batch in dataloader:
    images = batch[0]
    b, h, w, c = images.shape
    # convert to float32 and scale to [0, 1]
    images = images.float() / 255
    images = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

    images_aug = augmenter(images)
    images_aug_seq.append(images_aug)

images_aug_seq = torch.cat(images_aug_seq, dim=0)
images_aug_seq = (images_aug_seq.permute(0, 2, 3, 1) * 255).to(torch.uint8)

out_path = Path("assets/greenaug_mask.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)
write_video(out_path, images_aug_seq, 10)
