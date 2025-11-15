"""
Vanilla BLIP similarity-based frame selector for HERBench evaluation.

Selects frames based on BLIP Image-Text Matching (ITM) similarity scores.
Supports two modes:
A. On-the-fly computation: Computes similarities for each video (default)
B. Precomputed embeddings: Loads pre-computed frame embeddings from disk
"""

import logging
from pathlib import Path
from typing import List, Union, Optional, Dict
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

from .base_selector import BaseFrameSelector

logger = logging.getLogger(__name__)


class VanillaBLIPFrameSelector(BaseFrameSelector):
    """
    Vanilla BLIP similarity-based frame selector.

    Selects top-k frames with highest BLIP ITM similarity scores to the question.

    Modes:
    1. On-the-fly (default): Extracts frames and computes similarities in real-time
    2. Precomputed: Loads precomputed frame embeddings from disk (faster)
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-large-coco",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gpu_rank: int = 0,
        target_fps: float = 2.0,
        use_precomputed: bool = False,
        precomputed_dir: Optional[Path] = None,
        batch_size: int = 16,
    ):
        """
        Initialize vanilla BLIP frame selector.

        Args:
            model_name: HuggingFace BLIP model ID
            device: Device to run model on
            gpu_rank: GPU device rank for multi-GPU setup (0, 1, 2, etc.)
            target_fps: Target FPS for frame extraction (on-the-fly mode)
            use_precomputed: Whether to use precomputed embeddings
            precomputed_dir: Directory containing precomputed embeddings
            batch_size: Batch size for processing frames

        Precomputed embeddings format:
            Directory structure:
                <precomputed_dir>/
                    <video_id>/
                        frames.npy          # Frame indices [N]
                        embeddings.npy      # Frame embeddings [N, D]

            Loading example:
                frames_indices = np.load(precomputed_dir / video_id / "frames.npy")
                frame_embeddings = np.load(precomputed_dir / video_id / "embeddings.npy")
        """
        super().__init__(target_fps=target_fps)

        # Handle device with GPU rank
        if device == "cuda" and torch.cuda.is_available():
            self.device = f"cuda:{gpu_rank}"
        else:
            self.device = device

        self.model_name = model_name
        self.use_precomputed = use_precomputed
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.batch_size = batch_size

        # Load BLIP model for on-the-fly mode
        if not use_precomputed:
            logger.info(f"Loading BLIP model: {model_name} on device {self.device}")
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name, use_safetensors=True
            ).to(self.device)
            self.model.eval()
            logger.info(f"BLIP model loaded successfully on {self.device}")
        else:
            if not self.precomputed_dir:
                raise ValueError("precomputed_dir must be provided when use_precomputed=True")
            if not self.precomputed_dir.exists():
                raise ValueError(f"Precomputed directory not found: {self.precomputed_dir}")
            logger.info(f"Using precomputed embeddings from: {self.precomputed_dir}")
            self.processor = None
            self.model = None

        logger.info(
            f"Initialized Vanilla BLIP selector "
            f"(mode: {'precomputed' if use_precomputed else 'on-the-fly'})"
        )

    def select_frames(
        self,
        video_path: Union[str, Path],
        question: str,
        k: int = 16
    ) -> List[int]:
        """
        Select top-k frames with highest similarity to question.

        Args:
            video_path: Path to video file
            question: Question text
            k: Number of frames to select

        Returns:
            List of frame indices sorted in temporal order
        """
        if self.use_precomputed:
            return self._select_from_precomputed(video_path, question, k)
        else:
            return self._select_on_the_fly(video_path, question, k)

    def _select_on_the_fly(
        self,
        video_path: Union[str, Path],
        question: str,
        k: int
    ) -> List[int]:
        """
        Select frames by computing similarities on-the-fly.

        Args:
            video_path: Path to video file
            question: Question text
            k: Number of frames to select

        Returns:
            List of frame indices sorted in temporal order
        """
        video_path = Path(video_path)

        # Extract frames at target FPS
        frames, frame_indices = self._extract_frames_at_fps(video_path)

        if len(frames) == 0:
            logger.error(f"No frames extracted from {video_path}")
            return []

        # Compute similarities
        similarities = self._compute_similarities(frames, question)

        # Select top-k frames
        selected_indices = self._select_top_k(
            frame_indices, similarities, k
        )

        return selected_indices

    def _select_from_precomputed(
        self,
        video_path: Union[str, Path],
        question: str,
        k: int
    ) -> List[int]:
        """
        Select frames using precomputed embeddings.

        Args:
            video_path: Path to video file
            question: Question text
            k: Number of frames to select

        Returns:
            List of frame indices sorted in temporal order
        """
        video_path = Path(video_path)

        # Get video ID from path (filename without extension)
        video_id = video_path.stem

        # Load precomputed embeddings
        embedding_dir = self.precomputed_dir / video_id

        if not embedding_dir.exists():
            logger.error(
                f"Precomputed embeddings not found for video: {video_id} "
                f"at {embedding_dir}"
            )
            # Fallback to uniform sampling
            return self._fallback_uniform_sampling(video_path, k)

        try:
            frames_file = embedding_dir / "frames.npy"
            embeddings_file = embedding_dir / "embeddings.npy"

            if not frames_file.exists() or not embeddings_file.exists():
                logger.error(
                    f"Missing precomputed files for {video_id}: "
                    f"frames.npy or embeddings.npy"
                )
                return self._fallback_uniform_sampling(video_path, k)

            frame_indices = np.load(frames_file)
            frame_embeddings = np.load(embeddings_file)

            # Compute text embedding
            text_embedding = self._compute_text_embedding(question)

            # Compute similarities using dot product (assumes normalized embeddings)
            similarities = np.dot(frame_embeddings, text_embedding)

            # Select top-k frames
            selected_indices = self._select_top_k(
                frame_indices.tolist(), similarities, k
            )

            return selected_indices

        except Exception as e:
            logger.error(f"Error loading precomputed embeddings: {e}")
            return self._fallback_uniform_sampling(video_path, k)

    def _extract_frames_at_fps(
        self,
        video_path: Path
    ) -> tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from video at target FPS.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (frames, frame_indices)
                frames: List of RGB numpy arrays
                frame_indices: List of frame indices
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return [], []

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame step for target FPS
        frame_step = int(fps / self.target_fps) if fps > self.target_fps else 1

        frames = []
        frame_indices = []

        for frame_idx in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_idx)

        cap.release()

        logger.debug(
            f"Extracted {len(frames)} frames at {self.target_fps} FPS "
            f"from {video_path.name}"
        )

        return frames, frame_indices

    @torch.no_grad()
    def _compute_similarities(
        self,
        frames: List[np.ndarray],
        query: str
    ) -> np.ndarray:
        """
        Compute similarity scores between frames and query text.

        Args:
            frames: List of RGB numpy arrays
            query: Text query

        Returns:
            Array of similarity scores [N]
        """
        if len(frames) == 0:
            return np.array([])

        all_scores = []

        # Process frames in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]

            # Convert to PIL Images
            pil_images = [Image.fromarray(frame) for frame in batch_frames]

            # Prepare inputs
            inputs = self.processor(
                images=pil_images,
                text=[query] * len(pil_images),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**inputs)

            # Extract ITM scores
            if hasattr(outputs, 'itm_score'):
                logits = outputs.itm_score
                probs = torch.softmax(logits, dim=-1)[:, 1]  # "match" probability
            else:
                logger.error("Cannot extract ITM scores from BLIP output")
                probs = torch.ones(len(pil_images), device=self.device) * 0.5

            all_scores.append(probs.cpu())

        scores = torch.cat(all_scores, dim=0).numpy()
        return scores

    @torch.no_grad()
    def _compute_text_embedding(self, query: str) -> np.ndarray:
        """
        Compute text embedding for query (used with precomputed mode).

        Note: This requires the BLIP model to be loaded. If using precomputed mode,
        you may need to instantiate a minimal BLIP model just for text encoding.

        Args:
            query: Text query

        Returns:
            Text embedding [D]
        """
        # For precomputed mode, we assume text embeddings can be computed
        # using a lightweight BLIP model or cached
        # For now, return a simple implementation

        # Lazy load model if needed for text encoding in precomputed mode
        if self.model is None:
            logger.info("Loading BLIP model for text encoding")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(
                self.model_name, use_safetensors=True
            ).to(self.device)
            self.model.eval()

        inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get text features
        if hasattr(self.model, 'text_encoder'):
            text_outputs = self.model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
        else:
            text_outputs = self.model.text_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        # Get pooled features
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]

        # Normalize embedding
        text_embedding = torch.nn.functional.normalize(text_features, dim=-1)

        return text_embedding.cpu().numpy().squeeze()

    def _select_top_k(
        self,
        frame_indices: List[int],
        similarities: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Select top-k frames with highest similarity scores.

        Args:
            frame_indices: List of frame indices
            similarities: Array of similarity scores
            k: Number of frames to select

        Returns:
            List of selected frame indices sorted in temporal order
        """
        n_frames = len(frame_indices)

        if n_frames == 0:
            return []

        # Limit k to available frames
        k = min(k, n_frames)

        # Get top-k indices by similarity
        top_k_positions = np.argsort(similarities)[::-1][:k]

        # Get corresponding frame indices
        selected_frames = [frame_indices[pos] for pos in top_k_positions]

        # Sort by temporal order
        selected_frames = sorted(selected_frames)

        logger.debug(
            f"Selected {len(selected_frames)} frames with similarities: "
            f"min={similarities[top_k_positions].min():.3f}, "
            f"max={similarities[top_k_positions].max():.3f}, "
            f"mean={similarities[top_k_positions].mean():.3f}"
        )

        return selected_frames

    def _fallback_uniform_sampling(
        self,
        video_path: Path,
        k: int
    ) -> List[int]:
        """
        Fallback to uniform sampling if precomputed embeddings not available.

        Args:
            video_path: Path to video file
            k: Number of frames to select

        Returns:
            List of uniformly sampled frame indices
        """
        logger.warning("Falling back to uniform sampling")

        video_info = self.get_video_info(video_path)
        total_frames = video_info["total_frames"]

        if total_frames == 0:
            return []

        if k >= total_frames:
            return list(range(total_frames))

        step = total_frames / k
        return [int(i * step) for i in range(k)]
