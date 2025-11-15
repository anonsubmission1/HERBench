"""
HERBench PyTorch Dataset implementation.

Loads questions from task JSON files and provides them for evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HERBenchDataset(Dataset):
    """
    PyTorch Dataset for HERBench evaluation.

    Loads all questions from task JSON files in the tasks directory.
    """

    def __init__(
        self,
        data_dir: Path,
        tasks: Optional[List[str]] = None,
    ):
        """
        Initialize HERBench dataset.

        Args:
            data_dir: Root directory containing 'tasks' and 'videos' subdirectories
            tasks: List of task names to load (None = load all tasks)
        """
        self.data_dir = Path(data_dir)
        self.tasks_dir = self.data_dir / "tasks"
        self.videos_dir = self.data_dir / "videos"

        if not self.tasks_dir.exists():
            raise ValueError(f"Tasks directory not found: {self.tasks_dir}")

        if not self.videos_dir.exists():
            logger.warning(f"Videos directory not found: {self.videos_dir}")

        # Load questions
        self.questions = self._load_questions(tasks)

        logger.info(
            f"Loaded {len(self.questions)} questions from {len(self._get_unique_tasks())} tasks"
        )

    def _load_questions(self, task_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Load questions from task JSON files.

        Args:
            task_filter: List of task names to load (None = all)

        Returns:
            List of question dictionaries
        """
        questions = []

        # Find all JSON files in tasks directory
        json_files = sorted(self.tasks_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.tasks_dir}")
            return questions

        for json_file in json_files:
            task_name = json_file.stem

            # Filter by task name if specified
            if task_filter and task_name not in task_filter:
                continue

            try:
                with open(json_file, 'r') as f:
                    task_questions = json.load(f)

                # Add task file name to each question
                for q in task_questions:
                    q['task_file'] = task_name
                    # Ensure video_path is absolute or relative to data_dir
                    if 'video_path' in q:
                        video_path = Path(q['video_path'])
                        if not video_path.is_absolute():
                            q['video_path'] = str(self.data_dir / q['video_path'])

                questions.extend(task_questions)

                logger.info(f"Loaded {len(task_questions)} questions from {task_name}")

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        return questions

    def _get_unique_tasks(self) -> set:
        """Get set of unique task types in the dataset."""
        return set(q.get('task_type', q.get('task_file', 'unknown'))
                   for q in self.questions)

    def __len__(self) -> int:
        """Return number of questions in dataset."""
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get question at index.

        Args:
            idx: Question index

        Returns:
            Dictionary with question data:
                - question_id: Unique question identifier
                - question: Question text
                - candidates: List of answer candidates (with letter prefixes)
                - answer_choice: Correct answer letter (A-E)
                - answer_index: Correct answer index (0-4)
                - answer_text: Full answer text
                - video_path: Path to video file
                - task_type: Task category
                - metadata: Additional metadata
        """
        q = self.questions[idx]

        return {
            'question_id': q.get('question_id', f'Q{idx:04d}'),
            'question': q['question'],
            'candidates': q['candidates'],
            'answer_choice': q.get('answer_choice', 'A'),
            'answer_index': q.get('answer_index', 0),
            'answer_text': q.get('answer_text', ''),
            'video_path': q['video_path'],
            'video_id': q.get('video_id', ''),
            'task_type': q.get('task_type', q.get('task_file', 'unknown')),
            'metadata': q.get('metadata', {})
        }

    def get_task_statistics(self) -> Dict[str, int]:
        """
        Get question count per task type.

        Returns:
            Dictionary mapping task type to question count
        """
        task_counts = {}
        for q in self.questions:
            task_type = q.get('task_type', q.get('task_file', 'unknown'))
            task_counts[task_type] = task_counts.get(task_type, 0) + 1

        return task_counts


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Since we process one question at a time during evaluation,
    this simply returns the first (and only) item in the batch.

    Args:
        batch: List of question dictionaries

    Returns:
        Single question dictionary
    """
    if len(batch) == 1:
        return batch[0]
    else:
        # If batch size > 1, return as list
        return {
            'batch': batch
        }
