"""
InternVL3.5-8B model wrapper for HERBench evaluation.

Based on the official InternVL implementation.
Model: OpenGVLab/InternVL3_5-8B
"""

import logging
import math
from typing import List
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build image transformation pipeline."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess an image into multiple tiles.

    Args:
        image: PIL Image
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Size of each tile
        use_thumbnail: Whether to add a thumbnail

    Returns:
        List of preprocessed PIL Images
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class InternVL35_8BModel(BaseVLM):
    """InternVL3.5-8B model implementation."""

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3_5-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gpu_rank: int = 0,
        torch_dtype: str = "bfloat16",
        max_frames: int = 12,
        generation_config: dict = None,
        input_size: int = 448,
        max_num_tiles: int = 12,
    ):
        """
        Initialize InternVL3.5-8B model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            gpu_rank: GPU device rank for multi-GPU setup (0, 1, 2, etc.)
            torch_dtype: Data type for model weights
            max_frames: Maximum frames to process
            generation_config: Generation parameters
            input_size: Input image size (default 448)
            max_num_tiles: Maximum number of tiles per image (default 12)
        """
        super().__init__(
            model_id=model_id,
            device=device,
            gpu_rank=gpu_rank,
            torch_dtype=torch_dtype,
            max_frames=max_frames,
            generation_config=generation_config,
        )
        self.input_size = input_size
        self.max_num_tiles = max_num_tiles
        self.transform = build_transform(input_size=input_size)

    def _load_model(self):
        """Load InternVL3.5 model and tokenizer."""
        logger.info(f"Loading InternVL3.5-8B model: {self.model_id}")

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=None  # Manual device placement
        ).eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False
        )

        # Move model to device
        self.model = self.model.to(self.device)

        # InternVL doesn't use a separate processor
        self.processor = None

        logger.info("InternVL3.5-8B model loaded successfully")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image using dynamic preprocessing.

        Args:
            image: PIL Image

        Returns:
            Tensor of preprocessed image tiles [num_tiles, C, H, W]
        """
        # Apply dynamic preprocessing to get multiple tiles
        images = dynamic_preprocess(
            image,
            image_size=self.input_size,
            use_thumbnail=True,
            max_num=self.max_num_tiles
        )

        # Apply transform to each tile
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values

    def _prepare_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """
        Prepare frames ensuring they are PIL Images.

        Args:
            frames: Input frames

        Returns:
            Processed frames as PIL Images
        """
        # Limit to max_frames
        original_count = len(frames)
        if len(frames) > self.max_frames:
            step = len(frames) / self.max_frames
            frames = [frames[int(i * step)] for i in range(self.max_frames)]
            logger.info(f"Downsampled {original_count} frames to {len(frames)} (max_frames={self.max_frames})")
        else:
            logger.info(f"Using all {len(frames)} frames (max_frames={self.max_frames})")

        # Ensure frames are PIL Images in RGB mode
        processed_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                processed_frames.append(frame.convert('RGB'))
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[2] == 3:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    processed_frames.append(Image.fromarray(frame))
                else:
                    logger.warning(f"Invalid frame array shape: {frame.shape}")
                    processed_frames.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
            elif isinstance(frame, torch.Tensor):
                # Convert tensor to PIL Image
                if frame.dim() == 3:  # [C, H, W]
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                elif frame.dim() == 4 and frame.shape[0] == 1:  # [1, C, H, W]
                    frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                else:
                    frame_np = frame.cpu().numpy()

                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)

                if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                    processed_frames.append(Image.fromarray(frame_np))
                else:
                    processed_frames.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
            else:
                logger.warning(f"Unknown frame type: {type(frame)}")
                processed_frames.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))

        return processed_frames

    def _generate_with_frames(
        self,
        question: str,
        frames: List[Image.Image],
        candidates: List[str]
    ) -> str:
        """
        Generate response with visual frames.

        Args:
            question: Question text
            frames: List of PIL Images
            candidates: Answer candidates

        Returns:
            Generated response text
        """
        # Prepare frames
        frames = self._prepare_frames(frames)

        if not frames:
            logger.error("No frames available after preparation")
            return "ERROR_IN_PREDICTION"

        # Format prompt
        prompt = self._format_prompt(question, candidates)

        try:
            # Preprocess all frames and concatenate
            pixel_values_list = []
            for frame in frames:
                pixel_values = self._preprocess_image(frame)
                pixel_values_list.append(pixel_values)

            # Concatenate all frame tiles
            pixel_values = torch.cat(pixel_values_list, dim=0)
            pixel_values = pixel_values.to(self.torch_dtype).to(self.device)

            # Create video-style prompt with frame prefixes
            num_frames = len(frames)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
            full_prompt = video_prefix + prompt

            # Generate response using model.chat
            response, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                self.generation_config,
                history=None,
                return_history=True
            )

            # Clean up tensors
            del pixel_values, pixel_values_list
            torch.cuda.empty_cache()

            return response.strip()

        except Exception as e:
            logger.error(f"InternVL3.5-8B generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "ERROR_IN_PREDICTION"

    def _generate_without_frames(
        self,
        question: str,
        candidates: List[str]
    ) -> str:
        """
        Generate response without visual context (text-only).

        Args:
            question: Question text
            candidates: Answer candidates

        Returns:
            Generated response text
        """
        # Format prompt
        prompt = self._format_prompt(question, candidates)

        try:
            # Text-only generation (no pixel_values)
            response, history = self.model.chat(
                self.tokenizer,
                None,  # No pixel values for text-only
                prompt,
                self.generation_config,
                history=None,
                return_history=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"InternVL3.5-8B text-only generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "ERROR_IN_PREDICTION"
