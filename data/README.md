# HERBench Data

This directory contains the HERBench benchmark data.

## Directory Structure

```
data/
├── tasks/          # Task JSON files with questions
│   ├── temporal_shot_ordering.json
│   ├── multi_entities_grounding_and_localization.json
│   └── ...
└── videos/         # Video files organized by source
    ├── trailers/
    ├── HD_EPIC/
    ├── PersonPath22/
    └── WildTrack/
```

## Task JSON Format

Each task JSON file contains a list of questions with the following format:

```json
{
  "video_id": "Alien - Earth Official Trailer FX",
  "video_path": "videos/trailers/Alien - Earth Official Trailer FX.mp4",
  "question_id": "TSO_0000",
  "question": "The following 4 shots (scenes) take place in the video:\n\n...",
  "answer_text": "1->4->3->2",
  "answer_choice": "D",
  "answer_index": 3,
  "candidates": [
    "A. 1->2->3->4",
    "B. 4->1->3->2",
    "C. At least two descriptions...",
    "D. 1->4->3->2",
    "E. 1->4->2->3"
  ],
  "task_type": "Temporal Shot Ordering",
  "metadata": {
    "timestamps": ["00:08.0-00:09.0", "02:04.9-02:06.0", ...],
    "source_dataset": "trailers"
  }
}
```

### Fields Description

- **video_id**: Unique video identifier
- **video_path**: Relative path to video file from data directory
- **question_id**: Unique question identifier (e.g., TSO_0000)
- **question**: Question text
- **answer_text**: Full answer text
- **answer_choice**: Correct answer letter (A-E)
- **answer_index**: Zero-indexed position of correct answer (0-4)
- **candidates**: List of 5 answer choices with letter prefixes
- **task_type**: Task category name
- **metadata**: Additional information (timestamps, source dataset, etc.)

## Downloading Data

### Option 1: Using the download script

```bash
python scripts/download_data.py
```

### Option 2: Manual download

1. Download the task JSON files and place them in `data/tasks/`
2. Download the videos and organize them in `data/videos/` following the structure above
