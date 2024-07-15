import os

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
from einops import rearrange
from groundingdino.models import build_model
from groundingdino.util.inference import preprocess_caption
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from huggingface_hub import hf_hub_download
from torchvision.ops import box_convert
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
    SamModel,
    SamProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenerativeAugmentation(torch.nn.Module):
    def __init__(self, return_mask=False, device="cuda"):
        super().__init__()
        self.return_mask = return_mask
        self.device = device

        # Object Detector (Grounding DINO)
        self.grounding_dino_model = self._load_gd_model_hf(
            "ShilongLiu/GroundingDINO",
            "groundingdino_swint_ogc.pth",
            "GroundingDINO_SwinT_OGC.cfg.py",
        ).to(device)
        self.grounding_dino_transforms = v2.Compose(
            [
                v2.Resize(800, max_size=1333, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Segmenter (SAM)
        self.sam_processor = SamProcessor.from_pretrained(
            "facebook/sam-vit-base", do_rescale=False
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

        # Inpainter (Stable Diffusion + ControlNet)
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        ).to(device)
        self._feature_extractor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas", do_rescale=False
        )
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-depth-diffusers",
            # variant="fp16",
            use_safetensors=False,
            # torch_dtype=torch.float16,
        ).to(device)
        self.inpainter = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            controlnet=controlnet,
            # variant="fp16",
            use_safetensors=True,
            # torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        self.inpainter.scheduler = UniPCMultistepScheduler.from_config(
            self.inpainter.scheduler.config
        )

    def _load_gd_model_hf(self, repo_id, filename, ckpt_config_filename):
        cache_config_file = hf_hub_download(
            repo_id=repo_id, filename=ckpt_config_filename
        )

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location="cpu")
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        model.eval()
        return model

    def _get_depth_map(self, images):
        device = images.device
        images = self._feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            depth_map = self.depth_estimator(images).predicted_depth.to(device)

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

    def detect(
        self,
        images: torch.Tensor,
        caption: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        # From https://github.com/IDEA-Research/GroundingDINO/issues/102#issuecomment-1558728065
        images = self.grounding_dino_transforms(images)
        caption = preprocess_caption(caption=caption)

        outputs = self.grounding_dino_model(images, captions=[caption] * len(images))

        prediction_logits = outputs[
            "pred_logits"
        ].sigmoid()  # prediction_logits.shape = (bsz，nq, 256)
        prediction_boxes = outputs[
            "pred_boxes"
        ]  # prediction_boxes.shape = (bsz, nq, 4)

        logits_res = []
        boxs_res = []
        phrases_list = []
        tokenizer = self.grounding_dino_model.tokenizer
        for ub_logits, ub_boxes in zip(prediction_logits, prediction_boxes):
            mask = ub_logits.max(dim=1)[0] > box_threshold
            logits = ub_logits[mask]  # logits.shape = (n, 256)
            boxes = ub_boxes[mask]  # boxes.shape = (n, 4)

            tokenized = tokenizer(caption)
            phrases = [
                get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenizer
                ).replace(".", "")
                for logit in logits
            ]

            logits, indices = torch.sort(logits.max(dim=1)[0], descending=True)
            boxes = boxes[indices]
            phrases = [phrases[i] for i in indices]

            logits_res.append(logits)
            boxs_res.append(boxes)
            phrases_list.append(phrases)

        return boxs_res, logits_res, phrases_list

    def segment(self, images, boxes_xyxy, multimask_output=False):
        images = rearrange(images, "n c h w -> n h w c")
        device = images.device
        inputs = self.sam_processor(
            images, input_boxes=boxes_xyxy, return_tensors="pt"
        ).to(device)
        outputs = self.sam_model(**inputs, multimask_output=multimask_output)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores
        return masks, scores

    def inpaint(self, images, texts, masks):
        b, c, h, w = images.shape

        image_source_for_inpaint = resize(images, (512, 512), antialias=True)
        image_mask_for_inpaint = resize(masks, (512, 512), antialias=True)
        # image_mask_for_inpaint = (image_mask_for_inpaint * 255).to(torch.uint8)
        image_depth = self._get_depth_map(images)

        # image_source_for_inpaint = image_source_for_inpaint.to(torch.float32) / 255.0
        # image_mask_for_inpaint = image_mask_for_inpaint.to(torch.float32) / 255.0
        # image_depth = image_depth.to(torch.float32) / 255.0

        out = self.inpainter(
            prompt=texts,
            image=image_source_for_inpaint,
            mask_image=image_mask_for_inpaint,
            control_image=image_depth,
            num_inference_steps=4,
            controlnet_conditioning_scale=0.8,  # recommended for good generalization
            guidance_scale=0.0,
            output_type="pt",
        )

        # out = torch.as_tensor(out.images * 255, device=images.device, dtype=torch.uint8)
        # out = rearrange(out, "b h w c -> b c h w")
        out = out.images
        out = resize(out, (h, w), antialias=True)
        return out

    def forward(
        self,
        images,
        detect_text,  # One text prompt for all images
        inpaint_text,  # One text prompt per image
        detect_box_threshold=0.3,
        detect_text_threshold=0.25,
        sam_multimask_output=False,
    ):
        # First class in detect text is the appliance
        # images shape (b, h, w, c) uint8
        b, c, h, w = images.shape
        device = images.device
        # Detect with Grounding DINO
        boxess, logitss, phrasess = self.detect(
            images, detect_text, detect_box_threshold, detect_text_threshold
        )

        for i, boxes in enumerate(boxess):
            # https://github.com/pytorch/vision/issues/8258
            boxess[i] = (
                box_convert(
                    boxes * torch.tensor([w, h, w, h], device=device),
                    in_fmt="cxcywh",
                    out_fmt="xyxy",
                )
                .cpu()
                .tolist()
            )

        num_boxes_per_batch = [len(boxes) for boxes in boxess]
        max_num_boxes = max(num_boxes_per_batch)

        # SAM expects a fixed number of boxes per batch
        # Pad boxes with zeros to the maximum number of boxes in the batch
        padding_value = [0.0, 0.0, 0.0, 0.0]
        padded_boxess = [
            boxes + [padding_value] * (max_num_boxes - len(boxes)) for boxes in boxess
        ]

        # Segment with SAM
        padded_masks, padded_scores = self.segment(
            images,
            padded_boxess,
            multimask_output=sam_multimask_output,
        )
        # Remove padded masks
        masks = [
            _padded_mask[:_num_boxes]
            for _num_boxes, _padded_mask in zip(num_boxes_per_batch, padded_masks)
        ]

        # Combine masks of each object detected here
        # mask[:, -1] means taking the last mask of the multimask output
        background_masks = torch.stack(
            [mask[:, -1].sum(dim=0).bool() for mask in masks]
        )
        # Get background mask by inverting
        background_masks = 1 - background_masks.float()

        # Inpaint with SD
        out = self.inpaint(images, inpaint_text, background_masks.to(device))

        if self.return_mask:
            return out, background_masks
        return out
