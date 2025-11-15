# HERBench Usage Guide

Complete guide for running evaluations and configuring HERBench.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration System](#configuration-system)
- [Task Categories](#task-categories)
- [Model Configuration](#model-configuration)
- [Frame Selection Strategies](#frame-selection-strategies)
- [Adding New Models](#adding-new-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Advanced Usage](#advanced-usage)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation (15 minutes)

```bash
# Clone and setup
git clone https://github.com/your-username/HERBench.git
cd HERBench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
python scripts/download_data.py
```

### Basic Usage

#### 1. Run Evaluation (Simple)

```bash
# Qwen2.5-VL with uniform frame selection
python evaluation/run_evaluation.py model=qwen25vl frame_selector=uniform

# Results saved to: results/predictions_qwen25vl_uniform.json
```

#### 2. Calculate Accuracy

```bash
python evaluation/calculate_accuracy.py \
    --predictions results/predictions_qwen25vl_uniform.json
```

#### 3. Run MRFS Analysis

```bash
python evaluation/calculate_mrfs.py \
    model=qwen25vl \
    frame_selector=blip

# Results saved to: results/mrfs_qwen25vl_blip.json
```

### Common Configurations

#### Different Models

Currently implemented models:
- Qwen2.5-VL-7B-Instruct (`model=qwen25vl`)
- InternVL3.5-8B (`model=internvl35`)

```bash
# InternVL3.5-8B
python evaluation/run_evaluation.py model=internvl35 frame_selector=uniform

# Qwen2.5-VL
python evaluation/run_evaluation.py model=qwen25vl frame_selector=uniform
```

#### Different Frame Selectors

```bash
# Uniform sampling
python evaluation/run_evaluation.py frame_selector=uniform

# BLIP similarity (on-the-fly)
python evaluation/run_evaluation.py frame_selector=blip

# BLIP with precomputed embeddings
python evaluation/run_evaluation.py \
    frame_selector=blip \
    frame_selector.use_precomputed=true \
    frame_selector.precomputed_dir=/path/to/embeddings
```

#### Custom Settings

```bash
# Change number of frames
python evaluation/run_evaluation.py frame_selector.num_frames=32

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python evaluation/run_evaluation.py

# Custom data directory
python evaluation/run_evaluation.py data.data_dir=/path/to/data

# Evaluate specific tasks only
python evaluation/run_evaluation.py \
    data.tasks=[temporal_shot_ordering,multi_entities_grounding_and_localization]
```

### Output Files

After running evaluation:
- **Predictions**: `results/predictions_{model}_{selector}.json`
- **MRFS Results**: `results/mrfs_{model}_{selector}.json`
- **Accuracy Report**: Terminal output or saved to JSON with `--output` flag

## Configuration System

HERBench uses [Hydra](https://hydra.cc/) for configuration management.

### Configuration Files

- `configs/config.yaml` - Main evaluation configuration
- `configs/mrfs_config.yaml` - MRFS evaluation configuration
- `configs/model/*.yaml` - Model-specific configurations
- `configs/frame_selector/*.yaml` - Frame selector configurations

### Command Line Overrides

```bash
# Change data directory
python evaluation/run_evaluation.py data.data_dir=/path/to/data

# Modify model settings
python evaluation/run_evaluation.py \
    model=qwen25vl \
    model.max_frames=32 \
    model.torch_dtype=float16

# Use specific tasks
python evaluation/run_evaluation.py \
    data.tasks=[temporal_shot_ordering,multi_entities_grounding_and_localization]
```

## Task Categories

HERBench includes 12 tasks organized into 4 reasoning families:

### 1. Temporal Reasoning & Chronology [TR&C]

**Tasks:**
- **[TSO] Temporal Shot Ordering**: Arrange four shot descriptions from a trailer into the correct chronological order
- **[MPDR] Multi-Person Duration Reasoning**: Compare interval statistics for appearance-described people (e.g., who stayed longest)
- **[ASII] Action Sequence Integrity & Identification**: Select the correct ordering of five narrated egocentric actions

### 2. Referring & Tracking [R&T]

**Tasks:**
- **[AGBI] Appearance-Grounded Behavior Interactions**: Identify who accompanies or interacts with a uniquely described target
- **[AGAR] Appearance-Grounded Attribute Recognition**: Track a target to read out attributes anchored to their local context
- **[AGLT] Appearance-Grounded Localization Trajectory**: Recover path endpoints and coarse trajectory of a described target

### 3. Global Consistency & Verification [GC&V]

**Tasks:**
- **[FAM] False Action Memory**: Among several plausible actions, select the one that never occurs
- **[SVA] Scene Verification Arrangement**: Identify faithful shot descriptions and arrange them temporally, or abstain when too many are false
- **[FOM] False Object Memory**: Among plausible objects, identify the one the camera wearer does not interact with

### 4. Multi-Entity Aggregation & Numeracy [MEA&N]

**Tasks:**
- **[MEGL] Multi-Entities Grounding & Localization**: Given 2-3 appearance descriptions, decide which individuals actually appear
- **[AC] Action Counting**: Count occurrences of a specified action-object pair distributed across the timeline
- **[RLPC] Region-Localized People Counting**: Count unique individuals subject to spatial constraints

## Model Configuration

### Qwen2.5-VL Configuration

```yaml
# configs/model/qwen25vl.yaml
_target_: evaluation.model_wrappers.qwen25vl_7b.Qwen25VL7B
model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
max_frames: 16
torch_dtype: "bfloat16"
device_map: "auto"
use_direct_video: false
video_fps: 1.0
```

### InternVL3.5-8B Configuration

```yaml
# configs/model/internvl35.yaml
name: internvl35
model_id: "OpenGVLab/InternVL3_5-8B"
torch_dtype: "bfloat16"
max_frames: 16
generation_config:
  max_new_tokens: 50
  do_sample: true
```

### Direct Video Inference (Qwen2.5-VL)

For models that support direct video input:

```bash
pip install qwen-vl-utils

python evaluation/run_evaluation.py \
    model=qwen25vl \
    model.use_direct_video=true \
    model.video_fps=1.0
```

## Frame Selection Strategies

### Uniform Selection

Selects frames uniformly distributed across the video.

```bash
python evaluation/run_evaluation.py frame_selector=uniform
```

### BLIP Similarity-Based Selection

Selects frames with highest BLIP ITM similarity to the question.

**Mode A: On-the-fly computation** (default)
```bash
python evaluation/run_evaluation.py frame_selector=blip
```

**Mode B: Precomputed embeddings** (faster)
```bash
python evaluation/run_evaluation.py \
    frame_selector=blip \
    frame_selector.use_precomputed=true \
    frame_selector.precomputed_dir=/path/to/embeddings
```

Expected precomputed embeddings format:
```
<precomputed_dir>/
  <video_id>/
    frames.npy          # Frame indices [N]
    embeddings.npy      # Normalized frame embeddings [N, D]
```

## Adding New Models

To add a new model, create a wrapper class inheriting from `BaseVLM`:

```python
from evaluation.model_wrappers.base_vlm import BaseVLM

class YourModel(BaseVLM):
    def _load_model(self):
        # Load your model and processor
        self.model = ...
        self.processor = ...

    def _generate_with_frames(self, question, frames, candidates):
        # Generate response with visual context
        # frames: List of PIL Images
        # candidates: List of answer options
        return response_text

    def _generate_without_frames(self, question, candidates):
        # Generate response without visual context (text-only)
        return response_text
```

Then create a configuration file in `configs/model/your_model.yaml`:

```yaml
_target_: evaluation.model_wrappers.your_model.YourModel
model_name: "path/to/model"
max_frames: 16
torch_dtype: "bfloat16"
device_map: "auto"
# Add any model-specific parameters
```

## Evaluation Metrics

### Accuracy

Standard accuracy per task and overall:

```bash
python evaluation/calculate_accuracy.py --predictions results/predictions.json
```

**Output:**
- Overall accuracy
- Per-task accuracy
- Error statistics

### MRFS (Minimum Required Frame Set)

Measures the minimum number of frames needed for correct answers:

```bash
python evaluation/calculate_mrfs.py \
    model=qwen25vl \
    frame_selector=blip \
    mrfs.min_frames=1 \
    mrfs.max_frames=16
```

**MRFS values:**
- `0`: Text-only correct (no frames needed)
- `1-16`: Minimum frames required
- `"undefined"`: Incorrect even with maximum frames

## Advanced Usage

### Running on Specific Tasks

```bash
python evaluation/run_evaluation.py \
    data.tasks=[temporal_shot_ordering,false_action_memory]
```

### Batch Processing

```bash
# Process multiple models
for model in qwen25vl internvl35; do
    python evaluation/run_evaluation.py model=$model
done
```

### Custom Output Directory

```bash
python evaluation/run_evaluation.py \
    output_dir=/custom/path/results
```

## Data Format

### Predictions JSON

```json
{
  "TSO_0000": {
    "predicted_choice": "D",
    "predicted_index": 3,
    "is_correct": true,
    "response": "D. 1->4->3->2",
    "frames_used": 16,
    "gt_answer": "D",
    "full_answer": "D. 1->4->3->2",
    "task_type": "Temporal Shot Ordering"
  }
}
```

### MRFS Results JSON

```json
{
  "TSO_0000": {
    "mrfs": 8,
    "text_only_correct": false,
    "full_context_correct": true,
    "selected_frames": [12, 45, 78, 123, 156, 189, 234, 267],
    "binary_search_steps": 3,
    "question_id": "TSO_0000",
    "task_type": "Temporal Shot Ordering"
  }
}
```

### Repository Structure

```
HERBench/
├── data/
│   ├── tasks/              # Task JSON files
│   ├── videos/             # Video files
│   └── README.md
├── evaluation/
│   ├── model_wrappers/     # VLM implementations
│   │   ├── base_vlm.py
│   │   ├── qwenvl2_5_7b.py
│   │   └── internvl3_5_8b.py
│   ├── frame_selectors/    # Frame selection strategies
│   │   ├── base_selector.py
│   │   ├── uniform.py
│   │   └── vanila_blip_similarity.py
│   ├── herbench_dataset.py
│   ├── run_evaluation.py
│   ├── calculate_accuracy.py
│   └── calculate_mrfs.py
├── configs/                # Hydra configuration files
│   ├── config.yaml
│   ├── mrfs_config.yaml
│   ├── model/
│   └── frame_selector/
├── scripts/
│   └── download_data.py
├── requirements.txt
└── README.md
```

## Troubleshooting

### Out of Memory Errors

Reduce the number of frames or use float16:

```bash
python evaluation/run_evaluation.py \
    model.max_frames=8 \
    model.torch_dtype=float16
```

### Slow Frame Selection

Use precomputed embeddings for BLIP-based selection (see Frame Selection Strategies above).

### Model Loading Issues

Ensure you have the correct model weights and sufficient disk space:

```bash
# Check available space
df -h

# Clear Hugging Face cache if needed
rm -rf ~/.cache/huggingface/hub
```

### Missing Dependencies

```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade

# Install specific model dependencies
pip install qwen-vl-utils  # For Qwen2.5-VL
```

### Video Loading Errors

- Check video paths in task JSON files
- Ensure videos are in `data/videos/`
- Verify video file formats (MP4 recommended)

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Contact authors
- **Contributing**: See CONTRIBUTING.md
