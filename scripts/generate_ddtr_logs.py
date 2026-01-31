"""
Generate stochastic action logs from GTEA predictions for DDTR integration.

This script:
1. Loads a trained GTEA model
2. Runs inference on test videos
3. Converts binary predictions to action sequences
4. Exports in DDTR-compatible pickle format
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import GTEAHFFeatureDataset
from src.models import EventSegmentationModel


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def generate_predictions(model, loader, device="cpu"):
    """
    Generate frame-level predictions for all videos.
    
    Returns:
        predictions: List of binary predictions per video
        video_ids: List of video IDs
    """
    model.eval()
    predictions = []
    video_ids = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["features"].to(device)
            logits = model(inputs)  # [B, T, 1]
            probs = torch.sigmoid(logits).squeeze(-1)  # [B, T]
            mask = batch["mask"]  # [B, T]
            
            # Convert to binary predictions
            binary_preds = (probs > 0.5).long()  # [B, T]
            
            # Extract valid frames (non-padded)
            for b in range(binary_preds.shape[0]):
                valid_length = mask[b].sum().item()
                pred = binary_preds[b, :valid_length].cpu().numpy()
                predictions.append(pred)
                
                if "meta" in batch:
                    video_ids.append(batch["meta"]["video_id"][b])
                else:
                    video_ids.append(f"video_{len(predictions)-1}")
    
    return predictions, video_ids


def predictions_to_action_sequences(predictions, action_map=None):
    """
    Convert binary predictions (0=background, 1=event) to action sequences.
    
    This creates action IDs by detecting segments of consecutive 1s.
    - 0: background/no action
    - 1,2,3,...: different action instances
    
    Args:
        predictions: List of binary arrays
        action_map: Optional mapping of action IDs to names
        
    Returns:
        action_sequences: List of action ID sequences
    """
    action_sequences = []
    
    for pred in predictions:
        # Identify action boundaries
        changes = np.diff(np.concatenate([[0], pred, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        # Create action sequence: 0=background, 1+=action instances
        action_seq = np.zeros_like(pred)
        for action_id, (start, end) in enumerate(zip(starts, ends), start=1):
            action_seq[start:end] = action_id
        
        action_sequences.append(action_seq.reshape(-1, 1))
    
    return action_sequences


def create_ddtr_dataset(predictions, video_ids, output_path="gtea_ddtr.pkl"):
    """
    Create DDTR-compatible pickle file from predictions.
    
    DDTR format:
    {
        "target": [array1, array2, ...],        # Ground truth traces (same as stochastic for now)
        "stochastic": [array1, array2, ...]     # Stochastic/predicted traces
    }
    
    Args:
        predictions: List of action sequences (each is 2D array [T, 1])
        video_ids: List of video IDs (for reference)
        output_path: Where to save the pickle file
    """
    # For eventfulness segmentation, we use predictions as both target and stochastic
    # In a real scenario, you might generate slightly corrupted versions as stochastic
    dataset = {
        "target": predictions,
        "stochastic": predictions,
        "video_ids": video_ids,  # Extra metadata (optional)
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"✓ DDTR dataset saved to {output_path}")
    print(f"  - {len(predictions)} video sequences")
    print(f"  - Each sequence shape: [timesteps, 1]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DDTR-compatible stochastic logs from GTEA predictions"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config (e.g., configs/gtea_hf_train.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/gtea_hf/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="gtea_ddtr.pkl",
        help="Output path for DDTR pickle file",
    )
    parser.add_argument(
        "--cv_split",
        type=int,
        default=1,
        help="Cross-validation split (1-4)",
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
    print(f"Loading GTEA test set (cv_split={args.cv_split})...")
    ds = GTEAHFFeatureDataset(
        split="test",
        cv_split=args.cv_split,
        T=cfg["data"]["clip_len"],
        random_start=False,
    )
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Generate predictions
    print("Generating predictions...")
    predictions, video_ids = generate_predictions(model, loader, device)
    print(f"✓ Generated predictions for {len(predictions)} videos")

    # Convert to action sequences
    print("Converting to action sequences...")
    action_sequences = predictions_to_action_sequences(predictions)

    # Create DDTR dataset
    print("Creating DDTR-compatible dataset...")
    create_ddtr_dataset(action_sequences, video_ids, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("DDTR Log Generation Summary")
    print("=" * 60)
    print(f"Input: {len(predictions)} GTEA test videos")
    print(f"Output: {args.output}")
    print(f"Format: Python pickle with 'target' and 'stochastic' keys")
    print(f"Shape: Each sequence is [timesteps, 1] action indices")
    print("\nUsage with DDTR:")
    print(f"  1. Copy {args.output} to DDTR data directory")
    print(f"  2. Update DDTR config with data_path: {args.output}")
    print(f"  3. Run DDTR inference/training")
    print("=" * 60)


if __name__ == "__main__":
    main()
