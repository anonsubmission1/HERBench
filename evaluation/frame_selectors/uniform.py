"""
Uniform frame selector for HERBench evaluation.

Selects frames uniformly spaced throughout the video.
"""

import logging
from pathlib import Path
from typing import List, Union

from .base_selector import BaseFrameSelector

logger = logging.getLogger(__name__)


class UniformFrameSelector(BaseFrameSelector):
    """
    Uniform frame selector.

    Selects k frames uniformly distributed across the video duration.
    """

    def __init__(self, target_fps: float = 2.0):
        """
        Initialize uniform frame selector.

        Args:
            target_fps: Target FPS (not used for uniform selection, kept for interface compatibility)
        """
        super().__init__(target_fps=target_fps)
        logger.info("Initialized Uniform frame selector")

    def select_frames(
        self,
        video_path: Union[str, Path],
        question: str,
        k: int = 16
    ) -> List[int]:
        """
        Select k frames uniformly spaced throughout the video.

        Args:
            video_path: Path to video file
            question: Question text (not used for uniform selection)
            k: Number of frames to select

        Returns:
            List of frame indices sorted in temporal order
        """
        video_info = self.get_video_info(video_path)
        total_frames = video_info["total_frames"]

        if total_frames == 0:
            logger.error(f"Video has 0 frames: {video_path}")
            return []

        if k >= total_frames:
            # Return all frame indices if k is larger than total frames
            logger.warning(
                f"Requested {k} frames but video only has {total_frames} frames"
            )
            return list(range(total_frames))

        # Uniform sampling: divide video into k segments and take one frame from each
        step = total_frames / k
        frame_indices = [int(i * step) for i in range(k)]

        # Ensure we don't exceed bounds
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

        logger.debug(
            f"Selected {len(frame_indices)} frames uniformly from "
            f"{total_frames} total frames"
        )

        return frame_indices
