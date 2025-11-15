"""
Base frame selector interface for HERBench evaluation.

All frame selectors must inherit from this base class.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
import cv2

logger = logging.getLogger(__name__)


class BaseFrameSelector(ABC):
    """
    Abstract base class for frame selectors.

    Frame selectors determine which frames from a video should be used
    for answering a question.
    """

    def __init__(self, target_fps: float = 2.0):
        """
        Initialize frame selector.

        Args:
            target_fps: Target FPS for frame extraction (used by some selectors)
        """
        self.target_fps = target_fps

    @abstractmethod
    def select_frames(
        self,
        video_path: Union[str, Path],
        question: str,
        k: int = 16
    ) -> List[int]:
        """
        Select k frame indices from the video.

        Args:
            video_path: Path to video file
            question: Question text (may be used by some selectors)
            k: Number of frames to select

        Returns:
            List of frame indices (sorted in temporal order)
        """
        pass

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info: fps, total_frames, duration
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return {"fps": 30.0, "total_frames": 0, "duration": 0.0}

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {"fps": 30.0, "total_frames": 0, "duration": 0.0}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        }
