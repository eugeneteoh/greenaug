import argparse
from pathlib import Path
from urllib.parse import urlparse

import av
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import write_video

from greenaug import GreenAugGenerative, GreenAugRandom


def download_file(url, save_directory="."):
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    parsed_url = urlparse(url)
    local_filename = Path(parsed_url.path).name
    local_path = save_path / local_filename

    if not local_path.exists():
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"Downloaded: {local_path}")
    else:
        print(f"File already exists: {local_path}")


def main(args):
    video_url = "https://huggingface.co/datasets/eugeneteoh/greenaug/resolve/main/GreenScreenDemoCollection/open_drawer_green_screen.mp4"
    download_file(video_url, "data")

    # Load video
    container = av.open("data/open_drawer_green_screen.mp4")
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

    if args.augmenter == "GreenAugRandom":
        augmenter = GreenAugRandom()
    elif args.augmenter == "GreenAugGenerative":
        augmenter = GreenAugGenerative(device=args.device)
    elif args.augmenter == "GreenAugMask":
        pass
    elif args.augmenter == "GenerativeAugmentation":
        pass

    images_aug_seq = []
    for batch in dataloader:
        images = batch[0]
        b, h, w, c = images.shape
        # convert to float32 and scale to [0, 1]
        images = images.float() / 255
        images = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        extra_kwargs = {}
        if args.augmenter == "GreneAugGenerative":
            extra_kwargs.update(
                {
                    "prompt": [
                        "photorealistic kitchen, study room, washing room, living room, or bedroom"
                    ]
                    * b
                }
            )
        images_aug = augmenter(
            images, keycolor=["#439f82"] * b, tola=30, tolb=35, **extra_kwargs
        )
        images_aug_seq.append(images_aug)

    images_aug_seq = torch.cat(images_aug_seq, dim=0)
    images_aug_seq = (images_aug_seq.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    write_video("data/out.mp4", images_aug_seq, 10)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--augmenter", type=str, default="GreenAugRandom")
    argparser.add_argument("--device", type=str, default="cpu")
    args = argparser.parse_args()

    main(args)
