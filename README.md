# video-eventfulness-segmentation
Generic video event segmentation model with a shared ResNet + multi-scale temporal CNN backbone and task-specific heads.

## Tasks
- DDTR stochastic logs: per-frame categorical action distributions.
- Fall segmentation: per-frame binary logits (fall / no fall).

## Dataset manifest format
Provide `jsonl` manifests for train/val. Each line is a JSON object:
```
{"video_id": "vid_001", "frames_path": "path/to/frames.npy", "labels_path": "path/to/labels.npy"}
```
- `frames_path` expects `float32` arrays shaped `[T, C, H, W]` for `input_type: frames`.
- `features_path` can be used instead when `input_type: features`, shaped `[T, D]`.
- `labels_path` expects `int64` arrays shaped `[T]` (action ids for DDTR, 0/1 for fall).

## Training
```
python scripts/train.py --config configs/ddtr_train.yaml
python scripts/train.py --config configs/fall_train.yaml
```
