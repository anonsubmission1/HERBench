#!/usr/bin/env python3
"""
Download HERBench videos and tasks.

This script downloads:
1. Task JSON files to data/tasks/
2. Videos to data/videos/

The data is hosted on HuggingFace Datasets.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output_dir ./my_data
"""

import argparse
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_from_huggingface(output_dir: Path):
    """
    Download HERBench data from HuggingFace.

    Args:
        output_dir: Directory to save data
    """
    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading HERBench data from HuggingFace...")

        # TODO: Replace with actual HuggingFace dataset repository
        # repo_id = "your-username/HERBench"

        # For now, provide instructions
        logger.info(
            "\nHERBench data will be available on HuggingFace soon.\n"
            "Please contact the authors for access in the meantime.\n"
        )

        # Create directories
        tasks_dir = output_dir / "tasks"
        videos_dir = output_dir / "videos"

        tasks_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created data directories:")
        logger.info(f"  Tasks: {tasks_dir}")
        logger.info(f"  Videos: {videos_dir}")

        # TODO: Implement actual download
        # snapshot_download(
        #     repo_id=repo_id,
        #     repo_type="dataset",
        #     local_dir=output_dir,
        #     allow_patterns=["tasks/*.json", "videos/**/*.mp4"]
        # )

        logger.info("\nDirectory structure created. Awaiting data download implementation.")

    except ImportError:
        logger.error(
            "huggingface_hub not installed. "
            "Install with: pip install huggingface_hub"
        )
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download HERBench videos and tasks"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./data'),
        help='Output directory for data (default: ./data)'
    )

    args = parser.parse_args()

    logger.info(f"Output directory: {args.output_dir}")

    # Download data
    download_from_huggingface(args.output_dir)

    logger.info("\nDownload script completed!")
    logger.info(
        "\nNote: Please manually place your task JSON files in data/tasks/ "
        "and videos in data/videos/ until the HuggingFace dataset is available."
    )


if __name__ == "__main__":
    main()
