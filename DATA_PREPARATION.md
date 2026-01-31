# Fall Detection Data Preparation Guide

## Overview

This guide explains how to prepare your bus fall detection dataset for training.

---

## Step 1: Collect and Annotate Videos

### Video Requirements
- Format: MP4, AVI, MOV, etc. (any format readable by OpenCV)
- Frame rate: Any (script handles variable FPS)
- Resolution: Any (frames resized to ResNet input automatically)
- Duration: Variable (no fixed requirement)

### Annotation Format
Frame-level binary labels:
```
Frame 0:   0 (no fall)
Frame 1:   0 (no fall)
Frame 2:   1 (fall detected)
Frame 3:   1 (fall detected)
Frame 4:   1 (fall detected)
Frame 5:   0 (no fall)
...
```

### Annotation Tool
Use any tool that outputs per-frame binary labels:
- **CVAT** - Free, open-source video annotation
- **VGG Image Annotator** - Frame-by-frame labeling
- **Labelbox** - Professional platform
- **Custom script** - Simple Python script for in-house labeling

---

## Step 2: Extract Video Frames

### Option A: NumPy Array (Recommended)

```python
import cv2
import numpy as np
from pathlib import Path

def extract_video_frames(video_path, output_dir):
    """Extract all frames from video and save as NumPy array."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # BGR, [H, W, C]
    
    cap.release()
    
    frames = np.array(frames)  # [T, H, W, 3]
    
    # Save
    output_path = Path(output_dir) / f"{Path(video_path).stem}.npy"
    np.save(output_path, frames)
    
    print(f"✓ Saved {len(frames)} frames to {output_path}")
    print(f"  Shape: {frames.shape}")
    return frames

# Usage
extract_video_frames("videos/bus_001.mp4", "data/frames")
```

### Output
```
data/frames/
├── bus_001.npy    # [300, 720, 1280, 3]
├── bus_002.npy    # [450, 720, 1280, 3]
├── bus_003.npy    # [200, 720, 1280, 3]
└── ...
```

---

## Step 3: Load and Prepare Labels

### From Annotation File

**Option 1: CSV format**
```csv
video_id,frame,is_fall
bus_001,0,0
bus_001,1,0
bus_001,2,1
bus_001,3,1
```

```python
import pandas as pd

def load_labels_from_csv(csv_path, video_id):
    """Load labels for a specific video from CSV."""
    df = pd.read_csv(csv_path)
    video_data = df[df['video_id'] == video_id]
    labels = video_data['is_fall'].values.astype(np.int64)
    return labels

labels = load_labels_from_csv("annotations.csv", "bus_001")
np.save("data/labels/bus_001.npy", labels)
```

**Option 2: JSON format**
```json
{
  "bus_001": [0, 0, 1, 1, 1, 0, 0, ...],
  "bus_002": [0, 0, 0, 1, 0, ...],
  ...
}
```

```python
import json

def load_labels_from_json(json_path, video_id):
    """Load labels from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    labels = np.array(data[video_id], dtype=np.int64)
    return labels

labels = load_labels_from_json("annotations.json", "bus_001")
np.save("data/labels/bus_001.npy", labels)
```

**Option 3: VGG JSON (CVAT export)**
```python
def load_labels_from_vgg_json(vgg_json_path, video_id, num_frames):
    """Load labels from VGG Image Annotator JSON export."""
    with open(vgg_json_path) as f:
        data = json.load(f)
    
    labels = np.zeros(num_frames, dtype=np.int64)
    
    # Find entries for this video
    for img_id, regions in data.items():
        if video_id not in img_id:
            continue
        
        for region in regions['regions']:
            if region['region_attributes'].get('fall') == 'yes':
                # Extract frame number from image name
                frame_num = int(img_id.split('_')[-1])
                labels[frame_num] = 1
    
    return labels
```

### Verify Label Format

```python
def verify_labels(video_path, labels_path):
    """Check that labels match video length."""
    frames = np.load(video_path)
    labels = np.load(labels_path)
    
    assert len(frames) == len(labels), \
        f"Mismatch: {len(frames)} frames but {len(labels)} labels"
    
    print(f"✓ {len(frames)} frames")
    print(f"✓ {len(labels)} labels")
    print(f"✓ Fall ratio: {labels.mean():.1%}")
    
    # Check value range
    assert np.all((labels == 0) | (labels == 1)), \
        f"Invalid labels: must be 0 or 1, got {np.unique(labels)}"
    
    print("✓ All checks passed!")

verify_labels("data/frames/bus_001.npy", "data/labels/bus_001.npy")
```

---

## Step 4: Create Manifest Files

### Train/Val Split

```python
import json
from pathlib import Path

def create_manifest(video_dir, label_dir, split_ratio=0.8, seed=42):
    """Create train/val manifest files."""
    
    frames_dir = Path(video_dir)
    labels_dir = Path(label_dir)
    
    # List all videos
    frame_files = sorted(frames_dir.glob("*.npy"))
    
    # Create train/val split
    np.random.seed(seed)
    indices = np.random.permutation(len(frame_files))
    split_idx = int(len(frame_files) * split_ratio)
    
    train_files = [frame_files[i] for i in indices[:split_idx]]
    val_files = [frame_files[i] for i in indices[split_idx:]]
    
    # Write manifests
    def write_manifest(files, output_path):
        with open(output_path, 'w') as f:
            for frame_file in files:
                video_id = frame_file.stem
                label_file = labels_dir / f"{video_id}.npy"
                
                if not label_file.exists():
                    print(f"Warning: {label_file} not found, skipping")
                    continue
                
                entry = {
                    "video_id": video_id,
                    "frames_path": str(frame_file.absolute()),
                    "labels_path": str(label_file.absolute())
                }
                f.write(json.dumps(entry) + "\n")
        
        print(f"✓ {output_path} ({len(files)} videos)")
    
    write_manifest(train_files, "data/fall/train.jsonl")
    write_manifest(val_files, "data/fall/val.jsonl")

# Usage
create_manifest("data/frames", "data/labels")
```

### Output

**`data/fall/train.jsonl`:**
```jsonl
{"video_id": "bus_001", "frames_path": "/full/path/data/frames/bus_001.npy", "labels_path": "/full/path/data/labels/bus_001.npy"}
{"video_id": "bus_003", "frames_path": "/full/path/data/frames/bus_003.npy", "labels_path": "/full/path/data/labels/bus_003.npy"}
...
```

**`data/fall/val.jsonl`:**
```jsonl
{"video_id": "bus_002", "frames_path": "/full/path/data/frames/bus_002.npy", "labels_path": "/full/path/data/labels/bus_002.npy"}
{"video_id": "bus_004", "frames_path": "/full/path/data/frames/bus_004.npy", "labels_path": "/full/path/data/labels/bus_004.npy"}
...
```

---

## Step 5 (Alternative): Pre-extract Features

If you want faster training, pre-extract ResNet features:

```python
import torch
import torch.nn.functional as F
from src.models.backbone import ResNetFeatureExtractor
import numpy as np

def extract_features_batch(frames_npy_path, output_dir, batch_size=32):
    """Extract ResNet18 features from video frames."""
    
    # Load frames [T, H, W, 3]
    frames = np.load(frames_npy_path)
    
    # Initialize extractor
    extractor = ResNetFeatureExtractor(
        name="resnet18",
        pretrained=True,
        out_dim=512
    )
    extractor.eval()
    
    # Extract features in batches
    features_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]  # [B, H, W, 3]
            
            # Normalize and convert to tensor
            batch_tensor = torch.from_numpy(batch_frames).float() / 255.0
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # [B, 3, H, W]
            batch_tensor = batch_tensor.to(device)
            
            # Extract features
            batch_features = extractor(batch_tensor)  # [B, 512]
            features_list.append(batch_features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)  # [T, 512]
    
    # Save
    video_id = Path(frames_npy_path).stem
    output_path = Path(output_dir) / f"{video_id}.npy"
    np.save(output_path, features)
    
    print(f"✓ {output_path} - Shape: {features.shape}")
    return features

# Usage
extract_features_batch("data/frames/bus_001.npy", "data/features")
```

Then update manifests to use features:
```jsonl
{"video_id": "bus_001", "features_path": "data/features/bus_001.npy", "labels_path": "data/labels/bus_001.npy"}
```

And update config to use `input_type: features`.

---

## Complete Pipeline Example

```python
#!/usr/bin/env python
"""Complete fall detection data preparation pipeline."""

import cv2
import json
import numpy as np
from pathlib import Path

def process_all_videos(video_dir, annotation_file, output_dir):
    """
    Complete pipeline:
    1. Extract frames from videos
    2. Load annotations
    3. Verify and save
    4. Create manifests
    """
    
    # Create directories
    Path(f"{output_dir}/frames").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/labels").mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    
    for video_path in video_files:
        video_id = video_path.stem
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        frames = np.array(frames)
        
        # Get labels
        labels = np.array(annotations[video_id], dtype=np.int64)
        
        # Verify
        assert len(frames) == len(labels), \
            f"{video_id}: {len(frames)} frames but {len(labels)} labels"
        
        # Save
        np.save(f"{output_dir}/frames/{video_id}.npy", frames)
        np.save(f"{output_dir}/labels/{video_id}.npy", labels)
        
        print(f"✓ {video_id}: {len(frames)} frames, {labels.mean():.1%} falls")
    
    # Create manifests
    frame_files = sorted(Path(f"{output_dir}/frames").glob("*.npy"))
    
    indices = np.random.RandomState(42).permutation(len(frame_files))
    split = int(len(frame_files) * 0.8)
    
    for split_name, file_indices in [
        ("train", indices[:split]),
        ("val", indices[split:])
    ]:
        with open(f"{output_dir}/{split_name}.jsonl", 'w') as f:
            for idx in file_indices:
                video_id = frame_files[idx].stem
                entry = {
                    "video_id": video_id,
                    "frames_path": str(frame_files[idx].absolute()),
                    "labels_path": str(Path(f"{output_dir}/labels/{video_id}.npy").absolute())
                }
                f.write(json.dumps(entry) + "\n")
    
    print(f"\n✓ Train manifest: {output_dir}/train.jsonl")
    print(f"✓ Val manifest: {output_dir}/val.jsonl")

# Usage
process_all_videos(
    video_dir="path/to/videos",
    annotation_file="path/to/annotations.json",
    output_dir="data/fall"
)
```

---

## Summary Checklist

- [ ] Collect video files (MP4, AVI, etc.)
- [ ] Create frame-level binary annotations (0/1 per frame)
- [ ] Extract frames to NumPy arrays `[T, H, W, C]`
- [ ] Save labels as NumPy arrays `[T]`
- [ ] Create train/val JSONL manifests
- [ ] Run `python scripts/train.py --config configs/fall_detection_train.yaml`
- [ ] Evaluate with `python scripts/fall_detection_inference.py ...`

Done! Your dataset is ready for training.
