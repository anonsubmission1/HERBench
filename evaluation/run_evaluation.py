#!/usr/bin/env python3
"""
Run HERBench evaluation with a specified model and frame selector.

This script:
1. Loads the HERBench dataset
2. Initializes a VLM model and frame selector
3. Runs inference on all questions
4. Saves predictions to JSON file

Usage:
    # With Hydra config
    python run_evaluation.py model=qwen25vl frame_selector=uniform

    # With command line overrides
    python run_evaluation.py model=internvl35 frame_selector=blip \
        frame_selector.target_fps=1.0 evaluation.batch_size=1

    # Specify custom data directory
    python run_evaluation.py data.data_dir=/path/to/HERBench/data
"""

import json
import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

# Import HERBench components
from herbench_dataset import HERBenchDataset, collate_fn
from model_wrappers import Qwen25VL7BModel, InternVL35_8BModel
from frame_selectors import UniformFrameSelector, VanillaBLIPFrameSelector

logger = logging.getLogger(__name__)


def get_model(cfg: DictConfig):
    """
    Initialize model based on config.

    Args:
        cfg: Hydra config

    Returns:
        Initialized VLM model
    """
    model_name = cfg.model.name
    model_cfg = cfg.model

    logger.info(f"Initializing model: {model_name}")

    if model_name == "qwen25vl":
        model = Qwen25VL7BModel(
            model_id=model_cfg.model_id,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            torch_dtype=model_cfg.torch_dtype,
            max_frames=model_cfg.max_frames,
            generation_config=OmegaConf.to_object(model_cfg.generation_config),
            use_direct_video=model_cfg.get('use_direct_video', False),
            video_fps=model_cfg.get('video_fps', 1.0),
            video_max_pixels=model_cfg.get('video_max_pixels', 360 * 420)
        )
    elif model_name == "internvl35":
        model = InternVL35_8BModel(
            model_id=model_cfg.model_id,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            torch_dtype=model_cfg.torch_dtype,
            max_frames=model_cfg.max_frames,
            generation_config=OmegaConf.to_object(model_cfg.generation_config),
            max_num_tiles=model_cfg.get('max_num_tiles', 12)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Setup model
    model.setup()

    return model


def get_frame_selector(cfg: DictConfig):
    """
    Initialize frame selector based on config.

    Args:
        cfg: Hydra config

    Returns:
        Initialized frame selector
    """
    selector_name = cfg.frame_selector.name
    selector_cfg = cfg.frame_selector

    logger.info(f"Initializing frame selector: {selector_name}")

    if selector_name == "uniform":
        selector = UniformFrameSelector(
            target_fps=selector_cfg.target_fps
        )
    elif selector_name == "blip":
        selector = VanillaBLIPFrameSelector(
            model_name=selector_cfg.model_name,
            device=cfg.evaluation.device,
            gpu_rank=cfg.evaluation.get('gpu_rank', 0),
            target_fps=selector_cfg.target_fps,
            use_precomputed=selector_cfg.use_precomputed,
            precomputed_dir=selector_cfg.precomputed_dir,
            batch_size=selector_cfg.batch_size
        )
    else:
        raise ValueError(f"Unknown frame selector: {selector_name}")

    return selector


def run_evaluation(
    model,
    frame_selector,
    dataset: HERBenchDataset,
    cfg: DictConfig
) -> Dict:
    """
    Run evaluation on HERBench dataset.

    Args:
        model: VLM model
        frame_selector: Frame selector
        dataset: HERBench dataset
        cfg: Hydra config

    Returns:
        Dictionary mapping question_id to prediction results
    """
    predictions = {}

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one question at a time
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    logger.info(f"Starting evaluation on {len(dataset)} questions")

    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating", total=len(dataset))

    for question_data in pbar:
        question_id = question_data['question_id']

        try:
            # Select frames
            frame_indices = frame_selector.select_frames(
                video_path=question_data['video_path'],
                question=question_data['question'],
                k=cfg.frame_selector.num_frames
            )

            # Log frame count to verify config is respected
            logger.debug(f"Selected {len(frame_indices)} frames for {question_id}")

            # Clean up GPU memory after BLIP frame selection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not frame_indices:
                logger.warning(f"No frames selected for {question_id}")
                predictions[question_id] = {
                    'predicted_choice': 'ERROR',
                    'predicted_index': -1,
                    'is_correct': False,
                    'response': 'No frames available',
                    'frames_used': 0,
                    'gt_answer': question_data['answer_choice'],
                    'full_answer': question_data['candidates'][question_data['answer_index']],
                    'task_type': question_data['task_type']
                }
                continue

            # Run model prediction
            response = model.predict_with_frames(
                question=question_data['question'],
                video_path=question_data['video_path'],
                frame_indices=frame_indices,
                candidates=question_data['candidates']
            )

            # Extract answer
            predicted_choice = model.extract_answer_choice(response)

            # Calculate correctness
            gt_choice = question_data['answer_choice']
            is_correct = predicted_choice == gt_choice

            # Convert choice to index
            if predicted_choice in 'ABCDE':
                predicted_index = ord(predicted_choice) - ord('A')
            else:
                predicted_index = -1

            # Store prediction
            predictions[question_id] = {
                'predicted_choice': predicted_choice,
                'predicted_index': predicted_index,
                'is_correct': is_correct,
                'response': response,
                'frames_used': len(frame_indices),
                'gt_answer': gt_choice,
                'full_answer': question_data['candidates'][question_data['answer_index']],
                'task_type': question_data['task_type']
            }

            # Update progress bar
            accuracy = sum(1 for p in predictions.values() if p['is_correct']) / len(predictions)
            pbar.set_postfix({'acc': f'{accuracy:.3f}'})

        except Exception as e:
            logger.error(f"Error processing {question_id}: {e}")
            predictions[question_id] = {
                'predicted_choice': 'ERROR',
                'predicted_index': -1,
                'is_correct': False,
                'response': str(e),
                'frames_used': 0,
                'gt_answer': question_data['answer_choice'],
                'full_answer': question_data['candidates'][question_data['answer_index']],
                'task_type': question_data['task_type'],
                'error': str(e)
            }

    pbar.close()

    # Calculate final accuracy
    total = len(predictions)
    correct = sum(1 for p in predictions.values() if p['is_correct'])
    errors = sum(1 for p in predictions.values() if p['predicted_choice'] == 'ERROR')

    logger.info(f"\nEvaluation completed:")
    logger.info(f"  Total questions: {total}")
    logger.info(f"  Correct: {correct}")
    logger.info(f"  Errors: {errors}")

    # Calculate accuracy, handling case where all predictions are errors
    valid_predictions = total - errors
    if valid_predictions > 0:
        accuracy = correct / valid_predictions * 100
        logger.info(f"  Accuracy: {accuracy:.2f}%")
    else:
        logger.info(f"  Accuracy: N/A (all predictions resulted in errors)")

    return predictions


def save_predictions(predictions: Dict, output_file: Path):
    """
    Save predictions to JSON file.

    Args:
        predictions: Dictionary mapping question_id to prediction results
        output_file: Path to output JSON file
    """
    logger.info(f"Saving predictions to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info("Predictions saved successfully")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
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
    model = get_model(cfg)

    # Initialize frame selector
    frame_selector = get_frame_selector(cfg)

    # Run evaluation
    predictions = run_evaluation(model, frame_selector, dataset, cfg)

    # Save predictions
    output_file = Path(cfg.evaluation.output_dir) / f"predictions_{cfg.model.name}_{cfg.frame_selector.name}.json"
    save_predictions(predictions, output_file)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
