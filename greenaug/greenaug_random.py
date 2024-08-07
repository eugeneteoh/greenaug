import torch
from chromakey.torch import chroma_key


class GreenAugRandom(torch.nn.Module):
    def __init__(self, return_mask=False):
        super().__init__()
        self.return_mask = return_mask

    def forward(
        self,
        image,
        keycolor,
        background_image=None,
        tola=10,
        tolb=30,
        mask_threshold=None,
    ):
        image_out, mask = chroma_key(
            image,
            keycolor=keycolor,
            background_image=background_image,
            tola=tola,
            tolb=tolb,
        )
        if mask_threshold is not None:
            mask = (mask > mask_threshold).float()
            image_out = (image * mask[:, None, :, :]).to(torch.uint8)

        if self.return_mask:
            return image_out, mask
        return image_out
