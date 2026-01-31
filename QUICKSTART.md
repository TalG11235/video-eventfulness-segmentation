# Quick Start Guide

## Your Two Objectives

### ✅ Objective 1: Stochastic Logs for DDTR
Generate action segmentation logs from GTEA predictions for use in the DDTR project.

**3 simple steps:**

1. Train GTEA model (already done):
```bash
python scripts/train.py --config configs/gtea_hf_train.yaml
```

2. Generate DDTR logs:
```bash
python scripts/generate_ddtr_logs.py \
  --config configs/gtea_hf_train.yaml \
  --checkpoint outputs/gtea_hf/best.pt \
  --output gtea_ddtr.pkl
```

3. Use in DDTR:
```
Copy gtea_ddtr.pkl to DDTR project
Update DDTR config with: data_path: gtea_ddtr.pkl
Run DDTR training/inference
```

**That's it!** Your GTEA predictions are now ready for DDTR.

---

### ✅ Objective 2: Fall Detection in Buses

Full framework for training fall detection model on bus video data.

**Step-by-step:**

#### A. Prepare your data

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for complete guide:
- Extract frames from videos
- Create frame-level fall annotations (0=no fall, 1=fall)
- Generate manifest files (JSONL)

Expected structure:
```
data/fall/
├── frames/
│   ├── bus_001.npy    # [300 frames, H, W, 3]
│   ├── bus_002.npy
│   └── ...
├── labels/
│   ├── bus_001.npy    # [300 binary labels]
│   ├── bus_002.npy
│   └── ...
├── train.jsonl        # Manifest with paths
└── val.jsonl
```

#### B. Train model

```bash
python scripts/train.py --config configs/fall_detection_train.yaml
```

This will:
- Load your data from manifests
- Train for 50 epochs
- Save best model to `outputs/fall_detection/best.pt`
- Print loss/accuracy progress

#### C. Run inference

```bash
python scripts/fall_detection_inference.py \
  --config configs/fall_detection_train.yaml \
  --checkpoint outputs/fall_detection/best.pt \
  --manifest data/fall/test.jsonl \
  --output results.json
```

Outputs JSON with:
- Fall detected/not detected per video
- Exact frame timestamps
- Confidence scores
- Summary statistics

---

## Quick Reference

### Objective 1: DDTR Logs
```bash
# Step 1: Train GTEA (if not done)
python scripts/train.py --config configs/gtea_hf_train.yaml

# Step 2: Generate pickle for DDTR
python scripts/generate_ddtr_logs.py \
  --config configs/gtea_hf_train.yaml \
  --checkpoint outputs/gtea_hf/best.pt \
  --output gtea_ddtr.pkl
  
# Output: gtea_ddtr.pkl (ready for DDTR project)
```

### Objective 2: Fall Detection
```bash
# Step 1: Prepare data (see DATA_PREPARATION.md)
# - Extract frames to data/frames/
# - Create labels in data/labels/
# - Generate train.jsonl and val.jsonl

# Step 2: Train
python scripts/train.py --config configs/fall_detection_train.yaml

# Step 3: Inference
python scripts/fall_detection_inference.py \
  --config configs/fall_detection_train.yaml \
  --checkpoint outputs/fall_detection/best.pt \
  --manifest data/fall/test.jsonl \
  --output results.json
```

---

## Common Commands

```bash
# View GTEA training progress
tail -f outputs/gtea_hf/config.json

# Check fall detection data integrity
python -c "
from src.data.datasets import FallDetectionDataset
ds = FallDetectionDataset('data/fall/train.jsonl')
print(f'Train set: {len(ds)} videos')
for i in range(3):
    item = ds[i]
    print(f'  Video {i}: {item[\"meta\"][\"video_id\"]}, {item[\"meta\"][\"orig_len\"]} frames')
"

# Run on CPU if GPU not available
python scripts/train.py --config configs/fall_detection_train.yaml 2>&1 | head -20
# If it says "falling back to CPU", your GPU needs PyTorch CUDA setup

# Evaluate with lower threshold (more sensitive)
python scripts/fall_detection_inference.py \
  --config configs/fall_detection_train.yaml \
  --checkpoint outputs/fall_detection/best.pt \
  --manifest data/fall/test.jsonl \
  --threshold 0.3  # Lower = more sensitive
```

---

## Key Differences Between Tasks

| Feature | GTEA | Fall Detection |
|---------|------|-----------------|
| **Input** | Pre-extracted 2048-dim features | Raw frames OR features |
| **Data source** | HuggingFace (automatic) | Your own videos |
| **Annotation** | Built-in from HuggingFace | Manual per-frame labels |
| **Output** | DDTR pickle file | JSON with timestamps |
| **Training time** | ~1 hour (RTX 2080 Ti) | Depends on dataset size |
| **Use case** | Generate action logs | Detect fall events |

---

## Troubleshooting

### Issue: GPU out of memory
**Solution:** Reduce batch size in config
```yaml
training:
  batch_size: 4  # was 8
```

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution:** Run from project root directory
```bash
cd /path/to/video-eventfulness-segmentation
python scripts/train.py ...
```

### Issue: Manifest file not found
**Solution:** Use absolute paths in JSONL:
```jsonl
{"video_id": "bus_001", "frames_path": "/full/path/to/bus_001.npy", "labels_path": "/full/path/to/labels_001.npy"}
```

### Issue: Fall detection validation accuracy stuck at low value
**Solution:** This is normal on small datasets. Check loss - if it's decreasing, training is working.

---

## Documentation

- **[USAGE.md](USAGE.md)** - Comprehensive usage guide for all three tasks
- **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - Step-by-step data preparation for fall detection
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical details and architecture
- **[README.md](../README.md)** - Project overview

---

## Next Steps

1. **For DDTR:** Run the generate_ddtr_logs.py command above → copy pickle to DDTR project
2. **For Fall Detection:** 
   - Collect/annotate your bus video dataset (see DATA_PREPARATION.md)
   - Create manifest files
   - Run training
   - Evaluate on test set

Questions? See the detailed documentation files listed above.
