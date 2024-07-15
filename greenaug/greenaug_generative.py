import torch
from chromakey.torch import chroma_key
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
from torchvision.transforms.v2.functional import resize
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
)


class GreenAugGenerative(torch.nn.Module):
    def __init__(self, return_mask=False, device="cuda"):
        super().__init__()
        self.return_mask = return_mask
        self.device = device

        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        ).to(device)
        self._feature_extractor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas", do_rescale=False
        )
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-depth-diffusers",
            use_safetensors=False,
        ).to(device)
        self.inpainter = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            controlnet=controlnet,
            use_safetensors=True,
            safety_checker=None,
        ).to(device)
        self.inpainter.scheduler = UniPCMultistepScheduler.from_config(
            self.inpainter.scheduler.config
        )
        self.inpainter = self.inpainter.to(device)

    def _get_depth_map(self, images):
        device = images.device
        images = self._feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(images).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        images = torch.cat([depth_map] * 3, dim=1)

        return images

    def forward(
        self, image, keycolor, inpaint_text, tola=10, tolb=30, num_inference_steps=4
    ):
        b, c, h, w = image.shape

        image_out, mask = chroma_key(
            image,
            keycolor=keycolor,
            tola=tola,
            tolb=tolb,
        )

        image_source_for_inpaint = resize(image, (512, 512), antialias=True)
        image_mask_for_inpaint = resize(mask, (512, 512), antialias=True)
        image_depth = self._get_depth_map(image)

        image_source_for_inpaint = image_source_for_inpaint.float()
        image_mask_for_inpaint = image_mask_for_inpaint.float()
        image_depth = image_depth.float()

        image_out = self.inpainter(
            prompt=inpaint_text,
            image=image_source_for_inpaint,
            control_image=image_depth,
            mask_image=image_mask_for_inpaint,  # Use mask of background
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=0.8,
            guidance_scale=0.0,
            output_type="pt",
        )
        out = resize(image_out.images, (h, w), antialias=True)

        if self.return_mask:
            return out, mask
        return out
