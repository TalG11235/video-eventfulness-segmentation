# Video Eventfulness Segmentation - Usage Guide

## Overview

This project supports three tasks:

1. **GTEA Action Eventfulness** - Binary segmentation for kitchen action videos (eventful vs background)
2. **DDTR Stochastic Logs** - Generate action logs from GTEA predictions for DDTR project integration
3. **Fall Detection** - Binary segmentation for detecting falls in bus/transport videos

---

## Task 1: GTEA Action Eventfulness (✓ Ready)

### Training

```bash
python scripts/train.py --config configs/gtea_hf_train.yaml
```

**Config:** `configs/gtea_hf_train.yaml`
- Dataset: HuggingFace GTEA (pre-extracted 2048-dim features)
- Task: Binary eventfulness segmentation (0=background, 1=action)
- Cross-validation: Supports splits 1-4
- Output: Model checkpoints in `outputs/gtea_hf/`

**Data:**
- Train: 21 videos, 2,688 frames (88.2% eventful)
- Test: 7 videos, 896 frames (87.5% eventful)

### Inference & DDTR Log Generation

```bash
python scripts/generate_ddtr_logs.py \
  --config configs/gtea_hf_train.yaml \
  --checkpoint outputs/gtea_hf/best.pt \
  --output gtea_ddtr.pkl \
  --cv_split 1
```

**Output Format (DDTR-compatible):**
```python
{
    "target": [array1, array2, ...],      # Action sequences
    "stochastic": [array1, array2, ...],  # Stochastic traces
    "video_ids": ["vid_1", "vid_2", ...]  # Metadata
}
```

Each array shape: `[timesteps, 1]` with action IDs (0=background, 1+=action instances)

---

## Task 2: Fall Detection in Buses (New)

### Setup

First, prepare your fall detection dataset in JSONL manifest format:

**`data/fall/train.jsonl`:**
```jsonl
{"video_id": "bus_video_001", "features_path": "features/v001.npy", "labels_path": "labels/v001.npy"}
{"video_id": "bus_video_002", "features_path": "features/v002.npy", "labels_path": "labels/v002.npy"}
...
```

**`data/fall/val.jsonl`:**
```jsonl
{"video_id": "bus_video_100", "features_path": "features/v100.npy", "labels_path": "labels/v100.npy"}
...
```

#### Two input modes:

**Option A: Pre-extracted Features (Fast)**
```jsonl
{"video_id": "bus_001", "features_path": "data/features/v001.npy", "labels_path": "data/labels/v001.npy"}
```
- Features: NumPy array `[T, D]` (T=frames, D=feature_dim)
- Labels: NumPy array `[T]` with binary values (0=no fall, 1=fall)

**Option B: Raw Video Frames (Auto-extracts with ResNet)**
```jsonl
{"video_id": "bus_001", "frames_path": "data/videos/v001.npy", "labels_path": "data/labels/v001.npy"}
```
- Frames: NumPy array `[T, H, W, C]` (e.g., `[300, 480, 640, 3]`)
- Automatically extracts 512-dim ResNet features

### Training

```bash
python scripts/train.py --config configs/fall_detection_train.yaml
```

**Config:** `configs/fall_detection_train.yaml`
- Feature dim: 512 (ResNet18)
- Input type: "features" or "frames"
- Output: Checkpoints in `outputs/fall_detection/`

### Inference

```bash
python scripts/fall_detection_inference.py \
  --config configs/fall_detection_train.yaml \
  --checkpoint outputs/fall_detection/best.pt \
  --manifest data/fall/test.jsonl \
  --output fall_detection_results.json \
  --threshold 0.5
```

**Output:** `fall_detection_results.json`
```json
[
  {
    "video_id": "bus_001",
    "total_frames": 300,
    "fall_frames": 45,
    "fall_ratio": 0.15,
    "segments": [
      {
        "start_frame": 120,
        "end_frame": 165,
        "duration_frames": 45,
        "avg_confidence": 0.87
      }
    ],
    "predictions": [0.02, 0.05, ..., 0.91, ...]
  }
]
```

---

## Data Format Details

### Feature Extraction

If you have raw video files, extract features first:

```python
import numpy as np
from src.models.backbone import ResNetFeatureExtractor
import torch

# Load video frames [T, H, W, C]
frames = np.load('video.npy')
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # -> [T, C, H, W]

# Extract features
extractor = ResNetFeatureExtractor(name="resnet18", pretrained=True, out_dim=512)
with torch.no_grad():
    features = extractor(frames_tensor)  # [T, 512]

# Save
np.save('features.npy', features.numpy())
```

### Label Format

Binary labels (0 or 1) per frame:
```python
labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, ...])  # Shape: [T]
np.save('labels.npy', labels)
```

---

## Configuration Options

### Model Config
```yaml
model:
  input_type: features  # or "frames" for raw video
  feature_dim: 512      # ResNet18 output dimension
  temporal_dim: 256     # Hidden size in temporal layers
  ms_kernel_sizes: [3, 5, 7]   # Multi-scale kernel sizes
  ms_dilations: [1, 2, 3]      # Dilation rates
  dropout: 0.1
```

### Data Config
```yaml
data:
  dataset: fall_detection
  clip_len: 128              # Window size (frames)
  manifest_train: path/to/train.jsonl
  manifest_val: path/to/val.jsonl
  input_type: features       # or "frames"
  random_start: true         # Random window position during training
```

### Training Config
```yaml
training:
  batch_size: 8
  epochs: 50
  lr: 0.0001
  weight_decay: 0.0001
  num_workers: 0             # Must be 0 for HuggingFace datasets
  device: cuda               # Automatically falls back to CPU if not available
```

---

## Common Issues & Solutions

### Issue: Constant validation accuracy
**Cause:** Severe class imbalance (88% positive class)
**Solution:** Loss already handles masked frames. This is expected behavior on small datasets.

### Issue: "input_type features not recognized"
**Solution:** Use lowercase: `input_type: features` (not "Features")

### Issue: Out of memory on GPU
**Solution:** Reduce batch size in config: `batch_size: 4` or `batch_size: 2`

### Issue: num_workers > 0 causes errors with HuggingFace datasets
**Solution:** Set `num_workers: 0` in training config

---

## Checkpoints & Model Loading

Models are saved automatically during training:

- `outputs/gtea_hf/best.pt` - Best validation loss checkpoint
- `outputs/gtea_hf/last.pt` - Most recent checkpoint
- `outputs/gtea_hf/config.json` - Training config

### Load checkpoint for inference:

```python
import torch
from src.models import EventSegmentationModel

# Load config
import json
with open('outputs/gtea_hf/config.json') as f:
    cfg = json.load(f)

# Create model
model = EventSegmentationModel(
    task=cfg['task'],
    input_type=cfg['model']['input_type'],
    feature_dim=cfg['model']['feature_dim'],
    temporal_dim=cfg['model']['temporal_dim'],
)

# Load weights
checkpoint = torch.load('outputs/gtea_hf/best.pt')
model.load_state_dict(checkpoint['model_state'])
```

---

## Next Steps

1. **Prepare fall detection data:**
   - Collect bus video dataset
   - Extract features or prepare raw frames
   - Create JSONL manifest files

2. **Train fall detection model:**
   - Run training with your manifest files
   - Monitor loss/accuracy on validation set

3. **Generate DDTR logs:**
   - Train GTEA model (already done)
   - Run inference script to generate pickle files
   - Use in DDTR project

---

## Project Structure

```
src/
├── data/
│   ├── datasets/
│   │   ├── gtea_hf.py              # GTEA dataset loader
│   │   ├── fall_detection.py       # Fall detection dataset (NEW)
│   │   └── ...
│   └── base.py
├── models/
│   ├── model.py                    # Main model
│   ├── backbone.py                 # Feature extraction
│   └── heads.py                    # Classification heads

scripts/
├── train.py                         # Training script
├── generate_ddtr_logs.py           # GTEA→DDTR conversion (NEW)
└── fall_detection_inference.py     # Fall detection inference (NEW)

configs/
├── gtea_hf_train.yaml              # GTEA config
└── fall_detection_train.yaml       # Fall detection config (NEW)
```
