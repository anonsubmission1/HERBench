#!/usr/bin/env python3
"""
Calculate accuracy metrics from HERBench prediction results.

Takes a predictions JSON file and computes:
- Overall accuracy
- Per-task accuracy
- Detailed statistics

Usage:
    python calculate_accuracy.py --predictions results/predictions.json
    python calculate_accuracy.py --predictions results/predictions.json --output results/accuracy.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_predictions(predictions_file: Path) -> Dict:
    """
    Load predictions from JSON file.

    Args:
        predictions_file: Path to predictions JSON

    Returns:
        Dictionary mapping question_id to prediction data
    """
    logger.info(f"Loading predictions from: {predictions_file}")

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions")
    return predictions


def calculate_accuracy(predictions: Dict) -> Dict:
    """
    Calculate accuracy metrics from predictions.

    Args:
        predictions: Dictionary mapping question_id to prediction data

    Returns:
        Dictionary with accuracy metrics
    """
    # Group predictions by task
    task_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    overall_correct = 0
    overall_total = 0
    overall_errors = 0

    for question_id, pred in predictions.items():
        task_type = pred.get('task_type', 'unknown')

        # Check if prediction was correct
        is_correct = pred.get('is_correct', False)

        # Check for errors
        predicted_choice = pred.get('predicted_choice', '')
        is_error = predicted_choice == 'ERROR'

        # Update task statistics
        task_results[task_type]['total'] += 1
        overall_total += 1

        if is_error:
            task_results[task_type]['errors'] += 1
            overall_errors += 1
        elif is_correct:
            task_results[task_type]['correct'] += 1
            overall_correct += 1

    # Calculate per-task accuracy
    task_accuracy = {}
    for task, stats in task_results.items():
        total = stats['total']
        correct = stats['correct']
        errors = stats['errors']
        # Accuracy excludes errors
        valid_total = total - errors
        accuracy = (correct / valid_total * 100) if valid_total > 0 else 0.0

        task_accuracy[task] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors,
            'valid_total': valid_total
        }

    # Calculate overall accuracy
    valid_overall_total = overall_total - overall_errors
    overall_accuracy = (
        overall_correct / valid_overall_total * 100
        if valid_overall_total > 0 else 0.0
    )

    results = {
        'overall': {
            'accuracy': overall_accuracy,
            'correct': overall_correct,
            'total': overall_total,
            'errors': overall_errors,
            'valid_total': valid_overall_total
        },
        'per_task': task_accuracy
    }

    return results


def print_results(results: Dict):
    """
    Print accuracy results in a formatted way.

    Args:
        results: Dictionary with accuracy metrics
    """
    print("\n" + "=" * 70)
    print("HERBench Accuracy Results")
    print("=" * 70)

    # Overall accuracy
    overall = results['overall']
    print(f"\nOverall Accuracy: {overall['accuracy']:.2f}%")
    print(f"  Correct: {overall['correct']}/{overall['valid_total']}")
    if overall['errors'] > 0:
        print(f"  Errors: {overall['errors']} (excluded from accuracy)")
    print(f"  Total Questions: {overall['total']}")

    # Per-task accuracy
    print("\n" + "-" * 70)
    print("Per-Task Accuracy:")
    print("-" * 70)

    task_accuracy = results['per_task']
    sorted_tasks = sorted(task_accuracy.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for task, stats in sorted_tasks:
        print(f"\n{task}:")
        print(f"  Accuracy: {stats['accuracy']:.2f}%")
        print(f"  Correct: {stats['correct']}/{stats['valid_total']}")
        if stats['errors'] > 0:
            print(f"  Errors: {stats['errors']}")
        print(f"  Total Questions: {stats['total']}")

    print("\n" + "=" * 70 + "\n")


def save_results(results: Dict, output_file: Path):
    """
    Save accuracy results to JSON file.

    Args:
        results: Dictionary with accuracy metrics
        output_file: Path to output JSON file
    """
    logger.info(f"Saving results to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Calculate accuracy from HERBench prediction results"
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to save accuracy results JSON (optional)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.predictions.exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        return

    # Load predictions
    predictions = load_predictions(args.predictions)

    # Calculate accuracy
    results = calculate_accuracy(predictions)

    # Print results
    print_results(results)

    # Save results if output path specified
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
