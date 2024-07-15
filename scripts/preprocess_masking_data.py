import random
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from joblib import Parallel, delayed
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.io import read_video
from tqdm import tqdm

from greenaug import GreenAugRandom


class PreprocessMaskingDataset(Dataset):
    def __init__(self, images, background_root):
        self.images = images
        self.background_paths = sorted(Path(background_root).glob("**/*.png"))
        if not self.background_paths:
            raise ValueError("No chroma key background textures found.")

        self.__getitem__(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        h, w, c = image.shape

        bg_path = random.choice(self.background_paths)
        bg = Image.open(bg_path).convert("RGB")
        bg = bg.resize((w, h))
        bg = np.array(bg)

        image = rearrange(image, "h w c -> c h w")
        bg = rearrange(bg, "h w c -> c h w")

        image = image / 255.0
        bg = bg / 255.0

        return image, bg


def save_image(image, out_path):
    image = (
        image.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    image = Image.fromarray(image)
    image.save(out_path)


@hydra.main(version_base=None, config_path="../conf", config_name="mask")
def main(cfg):
    seed_everything(cfg.seed)

    preprocessed_root = Path(cfg.preprocessed_root)
    preprocessed_root.mkdir(exist_ok=True, parents=True)

    # dataset = PreprocessMattingDataset(
    #     cfg.input_root, cfg.background_root, cfg.camera_names
    # )
    # print(f"Original number of images: {len(dataset)}")
    # sampler = RandomSampler(dataset, replacement=True, num_samples=cfg.num_samples)
    # print(f"Number of images after preprocessing: {len(sampler)}")
    # dataloader = DataLoader(
    #     dataset,
    #     sampler=sampler,
    #     batch_size=cfg.batch_size,
    #     pin_memory=True,
    #     num_workers=args.num_workers,
    # )

    images = []
    video_paths = sorted(Path(cfg.raw_root).glob("**/*.mp4"))
    i = 0
    for path in video_paths:
        image_seq, _, _ = read_video(path)
        images.append(image_seq)
    images = torch.cat(images, dim=0)

    dataset = PreprocessMaskingDataset(images, cfg.background_root)
    sampler = RandomSampler(dataset, replacement=True, num_samples=cfg.num_samples)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    augmenter = GreenAugRandom(return_mask=True)

    i = 0
    for image, bg in tqdm(dataloader):
        image = image.to(cfg.device)
        bg = bg.to(cfg.device)
        b, c, h, w = image.shape

        out, mask = augmenter(
            image,
            keycolor=[cfg.keycolor] * b,
            tola=cfg.tola,
            tolb=cfg.tolb,
            background_image=bg,
        )
        mask = 1 - mask

        tasks = []
        for _image, _mask in zip(out, mask):
            _image = (
                _image.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            _image = Image.fromarray(_image)
            # _image.save(preprocessed_root / f"{i}.png")
            tasks.append(delayed(_image.save)(preprocessed_root / f"{i}.png"))

            _mask = (
                _mask.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
            )
            _mask = Image.fromarray(_mask)
            # _mask.save(preprocessed_root / f"{i}_mask.png")
            tasks.append(delayed(_mask.save)(preprocessed_root / f"{i}_mask.png"))
            i += 1
        n_jobs = cfg.num_workers if cfg.num_workers != 0 else 1
        Parallel(n_jobs=n_jobs)(tasks)


if __name__ == "__main__":
    main()
