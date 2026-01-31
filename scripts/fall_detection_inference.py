"""
Fall Detection Inference Script

Supports both:
1. Pre-extracted features (fast)
2. Raw video frames (extracts features on-the-fly with ResNet)

Outputs:
- Frame-level predictions
- Temporal fall segments
- JSON report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import FallDetectionDataset
from src.models import EventSegmentationModel


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def detect_falls(model, loader, device="cpu", confidence_threshold=0.5):
    """
    Generate fall detection predictions.
    
    Returns:
        results: List of detection results per video
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["features"].to(device)
            logits = model(inputs)  # [B, T, 1]
            probs = torch.sigmoid(logits).squeeze(-1)  # [B, T]
            mask = batch["mask"]  # [B, T]
            
            # Get predictions and confidence scores
            for b in range(probs.shape[0]):
                valid_length = mask[b].sum().item()
                pred_probs = probs[b, :valid_length].cpu().numpy()
                pred_binary = (pred_probs > confidence_threshold).astype(int)
                
                video_id = batch["meta"]["video_id"][b]
                
                # Find fall segments (consecutive 1s)
                segments = []
                in_segment = False
                start = 0
                
                for i in range(len(pred_binary)):
                    if pred_binary[i] == 1 and not in_segment:
                        start = i
                        in_segment = True
                    elif pred_binary[i] == 0 and in_segment:
                        segments.append({
                            "start_frame": int(start),
                            "end_frame": int(i),
                            "duration_frames": int(i - start),
                            "avg_confidence": float(pred_probs[start:i].mean())
                        })
                        in_segment = False
                
                # Handle case where fall extends to end
                if in_segment:
                    segments.append({
                        "start_frame": int(start),
                        "end_frame": int(len(pred_binary)),
                        "duration_frames": int(len(pred_binary) - start),
                        "avg_confidence": float(pred_probs[start:].mean())
                    })
                
                results.append({
                    "video_id": video_id,
                    "total_frames": int(valid_length),
                    "fall_frames": int(pred_binary.sum()),
                    "fall_ratio": float(pred_binary.mean()),
                    "segments": segments,
                    "predictions": pred_probs.tolist(),
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run fall detection inference on video dataset"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config (e.g., configs/fall_detection_train.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/fall_detection/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to test manifest (JSONL file)",
    )
    parser.add_argument(
        "--output",
        default="fall_detection_results.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for fall detection (0-1)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    # Check if device is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # Load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Create model
    print("Loading model...")
    model_cfg = cfg["model"]
    model = EventSegmentationModel(
        task=cfg["task"],
        input_type=model_cfg["input_type"],
        feature_dim=model_cfg["feature_dim"],
        temporal_dim=model_cfg.get("temporal_dim", 256),
        ms_kernel_sizes=tuple(model_cfg.get("ms_kernel_sizes", [3, 5, 7])),
        ms_dilations=tuple(model_cfg.get("ms_dilations", [1, 2, 3])),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    # Load checkpoint
    if Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        model = load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using randomly initialized model (for testing only)")

    # Create test dataset
    print(f"Loading test set from {args.manifest}...")
    ds = FallDetectionDataset(
        manifest_path=args.manifest,
        clip_len=cfg["data"]["clip_len"],
        random_start=False,
        input_type=cfg["model"]["input_type"],
    )
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Run inference
    print("Running inference...")
    results = detect_falls(model, loader, device, args.threshold)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Fall Detection Summary")
    print("=" * 60)
    total_videos = len(results)
    videos_with_falls = sum(1 for r in results if r["segments"])
    total_fall_segments = sum(len(r["segments"]) for r in results)
    avg_fall_ratio = np.mean([r["fall_ratio"] for r in results])
    
    print(f"Videos analyzed: {total_videos}")
    print(f"Videos with falls: {videos_with_falls} ({100*videos_with_falls/total_videos:.1f}%)")
    print(f"Total fall segments detected: {total_fall_segments}")
    print(f"Average fall ratio: {100*avg_fall_ratio:.1f}%")
    print(f"Confidence threshold: {args.threshold}")
    print("=" * 60)

    print("\nSample detections:")
    for result in results[:3]:
        print(f"\n{result['video_id']}:")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Fall frames: {result['fall_frames']} ({100*result['fall_ratio']:.1f}%)")
        if result["segments"]:
            for i, seg in enumerate(result["segments"]):
                print(f"  Fall {i+1}: frames {seg['start_frame']}-{seg['end_frame']}, "
                      f"confidence {seg['avg_confidence']:.3f}")
        else:
            print("  No falls detected")


if __name__ == "__main__":
    main()
