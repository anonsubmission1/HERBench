"""
Base Vision-Language Model interface for HERBench evaluation.

This module defines the abstract base class that all VLM implementations must inherit from.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models in HERBench.

    All model implementations must inherit from this class and implement:
    - _load_model(): Load the model and processor
    - _generate_with_frames(): Generate response with visual context
    - _generate_without_frames(): Generate response without visual context (text-only)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gpu_rank: int = 0,
        torch_dtype: str = "bfloat16",
        max_frames: int = 16,
        generation_config: Optional[Dict] = None,
    ):
        """
        Initialize base VLM.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run model on
            gpu_rank: GPU device rank for multi-GPU setup (0, 1, 2, etc.)
            torch_dtype: PyTorch dtype for model weights
            max_frames: Maximum number of frames to process
            generation_config: Generation configuration dict
        """
        self.model_id = model_id
        # Handle device with GPU rank
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_rank}")
        else:
            self.device = torch.device(device)

        # Handle torch dtype
        if isinstance(torch_dtype, str):
            if torch_dtype.lower() == "auto":
                self.torch_dtype = "auto"
            else:
                self.torch_dtype = getattr(torch, torch_dtype, torch.float32)
        else:
            self.torch_dtype = torch_dtype

        self.max_frames = max_frames

        # Default generation config
        self.generation_config = {
            "max_new_tokens": 50,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if generation_config:
            self.generation_config.update(generation_config)

        # Fix incompatible generation config: if do_sample=True, temperature must be > 0
        if self.generation_config.get("do_sample", False):
            if self.generation_config.get("temperature", 0.0) == 0.0:
                self.generation_config["temperature"] = 1.0
                logger.warning(
                    "do_sample=True requires temperature > 0. "
                    "Setting temperature=1.0 (was 0.0)"
                )

        # Will be set by _load_model()
        self.model = None
        self.processor = None
        self.tokenizer = None

        logger.info(f"Initializing {self.__class__.__name__} with model: {model_id}")

    def setup(self):
        """Setup model and processor."""
        logger.info(f"Loading model: {self.model_id}")
        self._load_model()
        logger.info(f"Model loaded successfully: {self.model_id}")

    @abstractmethod
    def _load_model(self):
        """
        Load model and processor. Must be implemented by subclasses.

        Should set:
        - self.model
        - self.processor
        - self.tokenizer (if different from processor)
        """
        pass

    @abstractmethod
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
            candidates: List of answer candidates

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def _generate_without_frames(
        self,
        question: str,
        candidates: List[str]
    ) -> str:
        """
        Generate response without visual context (text-only).

        Args:
            question: Question text
            candidates: List of answer candidates

        Returns:
            Generated response text
        """
        pass

    def predict_with_frames(
        self,
        question: str,
        video_path: Union[str, Path],
        frame_indices: List[int],
        candidates: List[str]
    ) -> str:
        """
        Make prediction with selected video frames.

        Args:
            question: Question text
            video_path: Path to video file
            frame_indices: List of frame indices to load
            candidates: List of answer candidates with letters (e.g., "A. answer1")

        Returns:
            Model's response text
        """
        try:
            # Load frames
            frames = self._load_frames_from_video(video_path, frame_indices)

            if not frames:
                logger.error(f"No frames loaded from {video_path}")
                return "ERROR_IN_PREDICTION"

            # Generate response
            response = self._generate_with_frames(question, frames, candidates)

            # Clean up
            self._cleanup_memory()

            return response

        except Exception as e:
            logger.error(f"Error in predict_with_frames: {e}")
            return "ERROR_IN_PREDICTION"

    def predict_without_frames(
        self,
        question: str,
        candidates: List[str]
    ) -> str:
        """
        Make prediction without visual context (text-only).

        Args:
            question: Question text
            candidates: List of answer candidates with letters

        Returns:
            Model's response text
        """
        try:
            response = self._generate_without_frames(question, candidates)
            self._cleanup_memory()
            return response
        except Exception as e:
            logger.error(f"Error in predict_without_frames: {e}")
            return "ERROR_IN_PREDICTION"

    def _load_frames_from_video(
        self,
        video_path: Union[str, Path],
        frame_indices: List[int]
    ) -> List[Image.Image]:
        """
        Load specific frames from video file.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract

        Returns:
            List of PIL Images
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []

        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in sorted(frame_indices):
            if idx < 0 or idx >= total_frames:
                logger.warning(f"Frame index {idx} out of range [0, {total_frames})")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                logger.warning(f"Could not read frame {idx}")

        cap.release()

        logger.debug(f"Loaded {len(frames)} frames from {video_path.name}")
        return frames

    def extract_answer_choice(self, response: str) -> str:
        """
        Extract answer choice (A-E) from model response.

        Args:
            response: Model's generated response

        Returns:
            Extracted letter (A-E) or 'ERROR' if not found
        """
        response = response.strip()

        # Check for error states first
        if 'ERROR' in response.upper() or not response:
            logger.warning(f"Error state detected in response: {response[:100]}")
            return 'ERROR'

        # Look for patterns like "A", "(A)", "A.", "Answer: A", etc.
        patterns = [
            r'^([A-E])[\s\.\,\)]',  # Letter at start with delimiter (space, period, comma, closing paren)
            r'^\(([A-E])\)',  # Letter in parentheses at start
            r'^([A-E])$',  # Just the letter alone
            r'[Aa]nswer:\s*([A-E])\b',  # "Answer: A" with word boundary
            r'[Cc]hoice:\s*([A-E])\b',  # "Choice: A"
            r'\b([A-E])\b[\.\,]',  # Single letter word followed by punctuation
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()

        # Look for standalone letter with word boundaries (not part of a word)
        # This ensures we don't match 'E' in 'ERROR' or 'A' in 'AVAILABLE'
        upper_response = response.upper()
        for match in re.finditer(r'([A-E])', upper_response):
            letter = match.group(1)
            start_idx = match.start(1)
            end_idx = match.end(1)

            # Check characters before and after to ensure it's standalone
            before_char = upper_response[start_idx - 1] if start_idx > 0 else ' '
            after_char = upper_response[end_idx] if end_idx < len(upper_response) else ' '

            # Letter is standalone if surrounded by non-letters
            before_ok = not before_char.isalpha()
            after_ok = not after_char.isalpha()

            if before_ok and after_ok:
                return letter

        logger.warning(f"Could not extract answer from: {response[:100]}")
        return 'ERROR'

    def _cleanup_memory(self):
        """Clean up GPU memory after inference."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _format_prompt(
        self,
        question: str,
        candidates: List[str]
    ) -> str:
        """
        Format prompt with question and candidates.

        Args:
            question: Question text
            candidates: List of answer candidates (already formatted with letters)

        Returns:
            Formatted prompt
        """
        prompt = f"{question}\n\n"

        # Check if candidates already have letters
        if candidates and candidates[0].strip()[0] in 'ABCDE':
            # Already formatted
            prompt += "\n".join(candidates)
        else:
            # Add letters
            for i, candidate in enumerate(candidates):
                letter = chr(65 + i)  # A, B, C, D, E
                prompt += f"{letter}. {candidate}\n"

        prompt += "\n\nPlease respond with only the correct answer letter (A, B, C, D, or E) without any explanations or additional text."
        return prompt


def get_model_cache_dir(model_id: str) -> Path:
    """
    Get local cache directory for a model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Path to cache directory
    """
    home_dir = Path.home()
    cache_dir = home_dir / "models_ckpts" / model_id.replace("/", "--")
    return cache_dir


def load_model_with_cache(
    model_id: str,
    model_class,
    processor_class,
    **kwargs
):
    """
    Load model and processor with local caching for faster loading.

    Args:
        model_id: HuggingFace model ID
        model_class: Model class to instantiate
        processor_class: Processor class to instantiate
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Tuple of (model, processor)
    """
    cache_dir = get_model_cache_dir(model_id)

    # Check if model exists in cache
    if cache_dir.exists() and (cache_dir / "config.json").exists():
        logger.info(f"Loading from cache: {cache_dir}")
        model_path = str(cache_dir)
    else:
        logger.info(f"Downloading model to cache: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_id
        kwargs['cache_dir'] = str(cache_dir.parent)

    # Load processor
    processor = processor_class.from_pretrained(
        model_path,
        trust_remote_code=kwargs.get('trust_remote_code', True)
    )

    # Load model
    model = model_class.from_pretrained(
        model_path,
        **kwargs
    )

    return model, processor
