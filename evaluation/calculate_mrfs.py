#!/usr/bin/env python3
"""
Calculate MRFS (Minimum Required Frame Set) for HERBench.

This script runs the MRFS evaluation end-to-end:
1. Loads HERBench dataset
2. Initializes model and frame selector
3. For each question, finds the minimum number of frames needed for correct answer
4. Saves MRFS results to JSON

Usage:
    # With Hydra config
    python calculate_mrfs.py model=qwen25vl frame_selector=blip

    # With command line overrides
    python calculate_mrfs.py model=internvl35 frame_selector=uniform \
        mrfs.min_frames=1 mrfs.max_frames=16
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Import HERBench components
from herbench_dataset import HERBenchDataset
from model_wrappers import Qwen25VL7BModel, InternVL35_8BModel
from frame_selectors import UniformFrameSelector, VanillaBLIPFrameSelector

logger = logging.getLogger(__name__)


class MRFSCalculator:
    """
    MRFS (Minimum Required Frame Set) calculator.

    Uses binary search to find the minimum number of frames needed
    for the model to correctly answer a question.
    """

    def __init__(
        self,
        model,
        frame_selector,
        min_frames: int = 1,
        max_frames: int = 16
    ):
        """
        Initialize MRFS calculator.

        Args:
            model: VLM model
            frame_selector: Frame selector
            min_frames: Minimum frames to test
            max_frames: Maximum frames to test
        """
        self.model = model
        self.frame_selector = frame_selector
        self.min_frames = min_frames
        self.max_frames = max_frames

    def compute_mrfs(self, question_data: Dict) -> Dict:
        """
        Compute MRFS for a single question.

        Algorithm:
        A. Test text-only (no frames) -> If correct, MRFS = 0
        B. Test with full context (max_frames) -> If incorrect, MRFS = undefined
        C. Binary search between min_frames and max_frames to find minimum

        Args:
            question_data: Question dictionary

        Returns:
            Dictionary with MRFS results:
                - mrfs: Number of frames (0, int, or "undefined")
                - text_only_correct: Boolean
                - full_context_correct: Boolean
                - selected_frames: Frame indices used
                - binary_search_steps: Number of search iterations
        """
        question = question_data['question']
        candidates = question_data['candidates']
        correct_choice = question_data['answer_choice']
        video_path = question_data['video_path']

        # Step A: Test text-only (no visual context)
        try:
            response = self.model.predict_without_frames(question, candidates)
            predicted_choice = self.model.extract_answer_choice(response)
            text_only_correct = predicted_choice == correct_choice

            if text_only_correct:
                return {
                    'mrfs': 0,
                    'text_only_correct': True,
                    'full_context_correct': None,
                    'selected_frames': [],
                    'binary_search_steps': 0
                }

        except Exception as e:
            logger.error(f"Error in text-only test: {e}")
            return {
                'mrfs': "error",
                'text_only_correct': False,
                'full_context_correct': None,
                'selected_frames': [],
                'binary_search_steps': 0,
                'error': str(e)
            }

        # Step B: Test with full visual context
        try:
            frame_indices = self.frame_selector.select_frames(
                video_path, question, k=self.max_frames
            )

            if not frame_indices:
                return {
                    'mrfs': "video_error",
                    'text_only_correct': False,
                    'full_context_correct': None,
                    'selected_frames': [],
                    'binary_search_steps': 0,
                    'error': "No frames extracted"
                }

            response = self.model.predict_with_frames(
                question, video_path, frame_indices, candidates
            )
            predicted_choice = self.model.extract_answer_choice(response)
            full_context_correct = predicted_choice == correct_choice

            if not full_context_correct:
                return {
                    'mrfs': "undefined",
                    'text_only_correct': False,
                    'full_context_correct': False,
                    'selected_frames': frame_indices,
                    'binary_search_steps': 0
                }

        except Exception as e:
            logger.error(f"Error in full context test: {e}")
            return {
                'mrfs': "error",
                'text_only_correct': False,
                'full_context_correct': None,
                'selected_frames': [],
                'binary_search_steps': 0,
                'error': str(e)
            }

        # Step C: Binary search for minimum frames
        min_k = self.min_frames
        max_k = self.max_frames
        best_k = self.max_frames
        best_frames = frame_indices
        search_steps = 0

        while min_k < max_k:
            search_steps += 1
            mid_k = (min_k + max_k) // 2

            if mid_k == min_k:
                break

            try:
                # Test with mid_k frames
                test_frames = self.frame_selector.select_frames(
                    video_path, question, k=mid_k
                )

                response = self.model.predict_with_frames(
                    question, video_path, test_frames, candidates
                )
                predicted_choice = self.model.extract_answer_choice(response)
                is_correct = predicted_choice == correct_choice

                if is_correct:
                    # Still correct with fewer frames
                    max_k = mid_k
                    best_k = mid_k
                    best_frames = test_frames
                else:
                    # Need more frames
                    min_k = mid_k + 1

            except Exception as e:
                logger.error(f"Error in binary search at k={mid_k}: {e}")
                break

        # Final verification
        try:
            final_frames = self.frame_selector.select_frames(
                video_path, question, k=min_k
            )
            response = self.model.predict_with_frames(
                question, video_path, final_frames, candidates
            )
            predicted_choice = self.model.extract_answer_choice(response)

            if predicted_choice == correct_choice:
                best_k = min_k
                best_frames = final_frames

        except Exception as e:
            logger.warning(f"Error in final verification: {e}")

        return {
            'mrfs': best_k,
            'text_only_correct': False,
            'full_context_correct': True,
            'selected_frames': best_frames,
            'binary_search_steps': search_steps
        }


def run_mrfs_evaluation(
    calculator: MRFSCalculator,
    dataset: HERBenchDataset,
    cfg: DictConfig
) -> Dict:
    """
    Run MRFS evaluation on dataset.

    Args:
        calculator: MRFS calculator
        dataset: HERBench dataset
        cfg: Hydra config

    Returns:
        Dictionary mapping question_id to MRFS results
    """
    results = {}

    logger.info(f"Starting MRFS evaluation on {len(dataset)} questions")

    pbar = tqdm(dataset, desc="MRFS Evaluation")

    for question_data in pbar:
        question_id = question_data['question_id']

        try:
            mrfs_result = calculator.compute_mrfs(question_data)

            results[question_id] = {
                **mrfs_result,
                'question_id': question_id,
                'task_type': question_data['task_type'],
                'video_path': question_data['video_path']
            }

            # Update progress bar with statistics
            numeric_mrfs = [
                r['mrfs'] for r in results.values()
                if isinstance(r['mrfs'], int)
            ]
            if numeric_mrfs:
                mean_mrfs = np.mean(numeric_mrfs)
                pbar.set_postfix({'mean_mrfs': f'{mean_mrfs:.1f}'})

        except Exception as e:
            logger.error(f"Error processing {question_id}: {e}")
            results[question_id] = {
                'mrfs': "error",
                'question_id': question_id,
                'task_type': question_data['task_type'],
                'error': str(e)
            }

    pbar.close()

    # Calculate summary statistics
    zero_mrfs = [r for r in results.values() if r['mrfs'] == 0]
    # MRFS: only count questions that needed frames (mrfs > 0)
    nonzero_mrfs = [r['mrfs'] for r in results.values() if isinstance(r['mrfs'], int) and r['mrfs'] > 0]
    undefined_mrfs = [r for r in results.values() if r['mrfs'] == "undefined"]
    error_mrfs = [r for r in results.values() if isinstance(r['mrfs'], str) and "error" in r['mrfs']]

    logger.info(f"\nMRFS Evaluation Summary:")
    logger.info(f"  Total questions: {len(results)}")
    logger.info(f"  Text-only (MRFS = 0): {len(zero_mrfs)} ({len(zero_mrfs)/len(results)*100:.1f}%)")
    logger.info(f"  Questions with MRFS: {len(nonzero_mrfs)} ({len(nonzero_mrfs)/len(results)*100:.1f}%)")
    if nonzero_mrfs:
        logger.info(f"    Mean MRFS: {np.mean(nonzero_mrfs):.2f}")
        logger.info(f"    Median MRFS: {np.median(nonzero_mrfs):.1f}")
        logger.info(f"    Min MRFS: {np.min(nonzero_mrfs)}")
        logger.info(f"    Max MRFS: {np.max(nonzero_mrfs)}")
    logger.info(f"  Undefined (incorrect with max frames): {len(undefined_mrfs)} ({len(undefined_mrfs)/len(results)*100:.1f}%)")
    logger.info(f"  Errors: {len(error_mrfs)} ({len(error_mrfs)/len(results)*100:.1f}%)")

    return results


def save_results(results: Dict, output_file: Path):
    """
    Save MRFS results to JSON file.

    Args:
        results: Dictionary mapping question_id to MRFS results
        output_file: Path to output JSON file
    """
    logger.info(f"Saving MRFS results to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("MRFS results saved successfully")


@hydra.main(version_base=None, config_path="../configs", config_name="mrfs_config")
def main(cfg: DictConfig):
    """Main MRFS evaluation function."""
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    if cfg.evaluation.seed is not None:
        torch.manual_seed(cfg.evaluation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.evaluation.seed)

    # Load dataset
    logger.info(f"Loading HERBench dataset from: {cfg.data.data_dir}")
    dataset = HERBenchDataset(
        data_dir=Path(cfg.data.data_dir),
        tasks=cfg.data.tasks if cfg.data.tasks else None
    )

    # Print dataset statistics
    task_stats = dataset.get_task_statistics()
    logger.info("Dataset statistics:")
    for task, count in sorted(task_stats.items()):
        logger.info(f"  {task}: {count} questions")

    # Initialize model
    model_name = cfg.model.name
    logger.info(f"Initializing model: {model_name}")

    if model_name == "qwen25vl":
        # IMPORTANT: MRFS calculation MUST use frame selector with ranking,
        # so we force use_direct_video=False regardless of config
        model = Qwen25VL7BModel(
            model_id=cfg.model.model_id,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            torch_dtype=cfg.model.torch_dtype,
            max_frames=cfg.model.max_frames,
            generation_config=OmegaConf.to_object(cfg.model.generation_config),
            use_direct_video=False,  # Always False for MRFS
        )
    elif model_name == "internvl35":
        model = InternVL35_8BModel(
            model_id=cfg.model.model_id,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            torch_dtype=cfg.model.torch_dtype,
            max_frames=cfg.model.max_frames,
            generation_config=OmegaConf.to_object(cfg.model.generation_config)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.setup()

    # Initialize frame selector
    selector_name = cfg.frame_selector.name
    logger.info(f"Initializing frame selector: {selector_name}")

    if selector_name == "uniform":
        frame_selector = UniformFrameSelector(
            target_fps=cfg.frame_selector.target_fps
        )
    elif selector_name == "blip":
        frame_selector = VanillaBLIPFrameSelector(
            model_name=cfg.frame_selector.model_name,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            target_fps=cfg.frame_selector.target_fps,
            use_precomputed=cfg.frame_selector.use_precomputed,
            precomputed_dir=cfg.frame_selector.precomputed_dir,
            batch_size=cfg.frame_selector.batch_size
        )
    else:
        raise ValueError(f"Unknown frame selector: {selector_name}")

    # Initialize MRFS calculator
    calculator = MRFSCalculator(
        model=model,
        frame_selector=frame_selector,
        min_frames=cfg.mrfs.min_frames,
        max_frames=cfg.mrfs.max_frames
    )

    # Run MRFS evaluation
    results = run_mrfs_evaluation(calculator, dataset, cfg)

    # Save results
    output_file = Path(cfg.evaluation.output_dir) / f"mrfs_{cfg.model.name}_{cfg.frame_selector.name}.json"
    save_results(results, output_file)

    logger.info("MRFS evaluation completed successfully!")


if __name__ == "__main__":
    main()
