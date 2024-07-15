import random
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as TV
from PIL import Image
from torch.utils.data import Dataset

plt.switch_backend("Agg")


class MaskingDataset(Dataset):
    def __init__(self, data_dir, apply_transforms=True):
        self.data_dir = Path(data_dir)
        self.image_indices = sorted(
            int(f.stem) for f in self.data_dir.glob("*.png") if "_mask" not in f.stem
        )
        self.apply_transforms = apply_transforms
        transforms = []
        if apply_transforms:
            transforms += [
                A.RandomResizedCrop(224, 224),
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        self.transforms = A.Compose(transforms)

        self.to_torch_image = TV.Compose(
            [
                TV.ToImage(),
                TV.ToDtype(torch.float32, scale=True),
            ]
        )
        self.__getitem__(0)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        image = Image.open(self.data_dir / f"{index}.png")
        mask = Image.open(self.data_dir / f"{index}_mask.png")

        transformed = self.transforms(image=np.array(image), mask=np.array(mask) / 255)

        data = {}
        data["image"] = self.to_torch_image(transformed["image"])
        data["mask"] = np.float32(transformed["mask"][None, :, :])
        return data


class MaskingNetwork(pl.LightningModule):
    def __init__(self, architecture, encoder_name, encoder_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            **kwargs,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name, encoder_weights)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self._dice_loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self._rng = random.Random()

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def predict(self, image, threshold=None):
        mask = self.forward(image).sigmoid()
        if threshold is not None:
            mask = mask >= threshold
        return mask

    def shared_step(self, batch, stage, batch_idx):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask >= 0.5).float()
        dice = self._dice_loss_fn(logits_mask, mask)
        # bce = F.binary_cross_entropy_with_logits(logits_mask, mask)
        mse = F.mse_loss(prob_mask, mask)
        bce = 0
        # mse = 0
        loss = dice + bce + mse
        # loss = dice

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")

        # if stage in ["val", "test"] and batch_idx == 0:
        #     figures = self.plot_results(image, mask, prob_mask)
        #     wandb.log(
        #         {
        #             **{f"{stage}/plot_{i}": fig for i, fig in enumerate(figures)},
        #             "trainer/global_step": self.trainer.global_step,
        #         }
        #     )
        #     plt.close("all")

        log_dict = {
            "loss": loss,
            "dice": dice,
            "bce": bce,
            "mse": mse,
            "accuracy": accuracy,
        }
        log_dict = {f"{stage}/{key}": val for key, val in log_dict.items()}
        self.log_dict(log_dict, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def plot_results(self, images, gt_masks, pr_masks, num_plots=5):
        random_indices = list(range(images.shape[0]))
        self._rng.shuffle(random_indices)
        random_indices = random_indices[:num_plots]

        images = images[random_indices].numpy(force=True)
        gt_masks = gt_masks[random_indices].numpy(force=True)
        pr_masks = pr_masks[random_indices].numpy(force=True)

        figs = []
        for image, gt_mask, pr_mask in zip(images, gt_masks, pr_masks):
            fig = plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image.transpose(1, 2, 0))  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(
                gt_mask.squeeze()
            )  # just squeeze classes dim, because we have only one class
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(
                pr_mask.squeeze()
            )  # just squeeze classes dim, because we have only one class
            plt.title("Prediction")
            plt.axis("off")
            figs.append(fig)

        return figs
