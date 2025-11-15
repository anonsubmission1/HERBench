"""
Qwen2.5-VL-7B-Instruct model wrapper for HERBench evaluation.

Based on the official HuggingFace implementation.
Model: Qwen/Qwen2.5-VL-7B-Instruct
"""

import logging
from pathlib import Path
from typing import List, Union
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from .base_vlm import BaseVLM, load_model_with_cache

logger = logging.getLogger(__name__)


class Qwen25VL7BModel(BaseVLM):
    """Qwen2.5-VL-7B-Instruct model implementation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gpu_rank: int = 0,
        torch_dtype: str = "bfloat16",
        max_frames: int = 16,
        generation_config: dict = None,
        use_direct_video: bool = False,
        video_fps: float = 1.0,
        video_max_pixels: int = 360 * 420,
    ):
        """
        Initialize Qwen2.5-VL-7B model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            gpu_rank: GPU device rank for multi-GPU setup (0, 1, 2, etc.)
            torch_dtype: Data type for model weights
            max_frames: Maximum frames to process (for frame-based inference)
            generation_config: Generation parameters
            use_direct_video: If True, feed video file directly instead of frames
            video_fps: Frames per second when using direct video (default: 1.0)
            video_max_pixels: Maximum pixels per frame for direct video (default: 360*420)
        """
        super().__init__(
            model_id=model_id,
            device=device,
            gpu_rank=gpu_rank,
            torch_dtype=torch_dtype,
            max_frames=max_frames,
            generation_config=generation_config,
        )
        self.use_direct_video = use_direct_video
        self.video_fps = video_fps
        self.video_max_pixels = video_max_pixels

    def _load_model(self):
        """Load Qwen2.5-VL model and processor."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_id}")

        self.model, self.processor = load_model_with_cache(
            self.model_id,
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=None,  # Manual device placement
        )

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Qwen2.5-VL model loaded successfully")

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
        # Limit frames to max_frames
        if len(frames) > self.max_frames:
            # Uniform sampling
            step = len(frames) / self.max_frames
            frames = [frames[int(i * step)] for i in range(self.max_frames)]

        # Format prompt
        prompt = self._format_prompt(question, candidates)

        # Qwen2.5-VL format: multiple images in conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in frames
                ] + [{"type": "text", "text": prompt}]
            }
        ]

        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=frames,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs['input_ids'], outputs)
            ]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Clean up tensors
            del inputs, outputs
            torch.cuda.empty_cache()

            return response.strip()

        except Exception as e:
            logger.error(f"Error in Qwen2.5-VL generation: {e}")
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

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs (no images)
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs['input_ids'], outputs)
            ]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Clean up tensors
            del inputs, outputs
            torch.cuda.empty_cache()

            return response.strip()

        except Exception as e:
            logger.error(f"Error in Qwen2.5-VL text-only generation: {e}")
            return "ERROR_IN_PREDICTION"

    def _generate_with_direct_video(
        self,
        question: str,
        video_path: Union[str, Path],
        candidates: List[str]
    ) -> str:
        """
        Generate response using direct video file (not extracted frames).

        Args:
            question: Question text
            video_path: Path to video file
            candidates: Answer candidates

        Returns:
            Generated response text
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            logger.error("qwen_vl_utils not installed. Install with: pip install qwen-vl-utils")
            return "ERROR_IN_PREDICTION"

        # Format prompt
        prompt = self._format_prompt(question, candidates)

        # Convert path to file:// URI
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return "ERROR_IN_PREDICTION"

        video_uri = f"file://{video_path.absolute()}"

        # Create messages with video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_uri,
                        "max_pixels": self.video_max_pixels,
                        "fps": self.video_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # Build text prompt from the multi-modal conversation
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Let qwen-vl-utils turn the messages into image/video tensors
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )

            # Processor merges text + vision into model inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            # Remove input tokens from output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs['input_ids'], outputs)
            ]

            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Clean up tensors
            del inputs, outputs, generated_ids
            if image_inputs:
                del image_inputs
            if video_inputs:
                del video_inputs
            torch.cuda.empty_cache()

            return response.strip()

        except Exception as e:
            logger.error(f"Error in Qwen2.5-VL direct video generation: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "ERROR_IN_PREDICTION"

    def predict_with_frames(
        self,
        question: str,
        video_path: Union[str, Path],
        frame_indices: List[int],
        candidates: List[str]
    ) -> str:
        """
        Make prediction with selected video frames or direct video.

        Args:
            question: Question text
            video_path: Path to video file
            frame_indices: List of frame indices to load (ignored if use_direct_video=True)
            candidates: List of answer candidates with letters (e.g., "A. answer1")

        Returns:
            Model's response text
        """
        try:
            # Use direct video mode if enabled
            if self.use_direct_video:
                logger.info(f"Using direct video inference for {video_path}")
                response = self._generate_with_direct_video(question, video_path, candidates)
            else:
                # Use standard frame-based inference
                frames = self._load_frames_from_video(video_path, frame_indices)

                if not frames:
                    logger.error(f"No frames loaded from {video_path}")
                    return "ERROR_IN_PREDICTION"

                response = self._generate_with_frames(question, frames, candidates)

            # Clean up
            self._cleanup_memory()

            return response

        except Exception as e:
            logger.error(f"Error in predict_with_frames: {e}")
            return "ERROR_IN_PREDICTION"
