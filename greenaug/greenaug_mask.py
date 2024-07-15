import torch
from torchvision.transforms.v2.functional import resize

from .masking_network import MaskingNetwork


class GreenAugMask(torch.nn.Module):
    def __init__(self, checkpoint=None, return_mask=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.return_mask = return_mask

        if checkpoint is not None:
            self.masking_network = MaskingNetwork.load_from_checkpoint(checkpoint)
        else:
            self.masking_network = MaskingNetwork(
                architecture="unet", encoder_name="resnet18", encoder_weights="imagenet"
            )

    def forward(self, image):
        b, c, h, w = image.shape
        image_in = resize(image, (224, 224), antialias=True)
        mask = self.masking_network.predict(image_in)
        out = image_in * mask
        out = resize(out, (h, w), antialias=True)
        if self.return_mask:
            return out, mask
        return out
