from pathlib import Path

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import write_video

from greenaug import GreenAugRandom

video_path = hf_hub_download(
    repo_id="eugeneteoh/greenaug",
    repo_type="dataset",
    filename="GreenScreenDemoCollection/open_drawer_green_screen.mp4",
)

background_root = snapshot_download(
    repo_id="eugeneteoh/mil_data", repo_type="dataset", allow_patterns="*.png"
)
background_root = Path(background_root)

# Load video
container = av.open(video_path)
frames = []
for i, frame in enumerate(container.decode(video=0)):
    if i > 10:
        break
    img = frame.to_ndarray(format="rgb24")
    frames.append(img)
container.close()
frames = np.asarray(frames)  # (T, H, W, C)

dataset = TensorDataset(torch.as_tensor(frames))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

augmenter = GreenAugRandom()

images_aug_seq = []
for batch in dataloader:
    images = batch[0]
    b, h, w, c = images.shape
    # convert to float32 and scale to [0, 1]
    images = images.float() / 255
    images = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

    images_aug = augmenter(images, keycolor=["#439f82"] * b, tola=30, tolb=35)
    images_aug_seq.append(images_aug)

images_aug_seq = torch.cat(images_aug_seq, dim=0)
images_aug_seq = (images_aug_seq.permute(0, 2, 3, 1) * 255).to(torch.uint8)

out_path = Path("assets/greenaug_random.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)
write_video(out_path, images_aug_seq, 10)
