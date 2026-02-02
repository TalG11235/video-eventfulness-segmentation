import json
import numpy as np
import pickle
from pathlib import Path

def f1_score(pred, gt, overlap_threshold):
    """Compute F1 score for temporal action segmentation."""
    pred = np.array(pred)
    gt = np.array(gt)

    # Find change points
    pred_changes = np.where(pred[:-1] != pred[1:])[0] + 1
    pred_changes = np.concatenate([[0], pred_changes, [len(pred)]])

    gt_changes = np.where(gt[:-1] != gt[1:])[0] + 1
    gt_changes = np.concatenate([[0], gt_changes, [len(gt)]])

    pred_segments = []
    for i in range(len(pred_changes) - 1):
        start, end = pred_changes[i], pred_changes[i + 1]
        pred_segments.append((start, end, pred[start]))

    gt_segments = []
    for i in range(len(gt_changes) - 1):
        start, end = gt_changes[i], gt_changes[i + 1]
        gt_segments.append((start, end, gt[start]))

    # Compute IoU for each pair
    tp = 0
    for pred_seg in pred_segments:
        pred_start, pred_end, pred_class = pred_seg
        for gt_seg in gt_segments:
            gt_start, gt_end, gt_class = gt_seg
            if pred_class != gt_class:
                continue
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
            if union > 0 and intersection / union >= overlap_threshold:
                tp += 1
                break

    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1

def main():
    # Load predictions
    with open('test_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)

    # Load test manifest
    test_items = []
    with open('data/50salads/test.split2.jsonl', 'r') as f:
        for line in f:
            test_items.append(json.loads(line.strip()))

    # Sort both by video_id to ensure alignment
    pred_video_ids = pred_data['video_ids']
    pred_indices = sorted(range(len(pred_video_ids)), key=lambda i: pred_video_ids[i])

    test_video_ids = [item['video_id'] for item in test_items]
    test_indices = sorted(range(len(test_video_ids)), key=lambda i: test_video_ids[i])

    # Compute F1 scores
    f1_10_scores = []
    f1_25_scores = []
    f1_50_scores = []

    for pred_idx, test_idx in zip(pred_indices, test_indices):
        video_id = pred_video_ids[pred_idx]
        assert video_id == test_video_ids[test_idx], f"Mismatch: {video_id} vs {test_video_ids[test_idx]}"

        # Load ground truth
        gt_path = Path(test_items[test_idx]['labels_path'])
        gt = np.load(gt_path)

        # Get predictions (argmax of probs)
        probs = pred_data['probs'][pred_idx]
        length = pred_data['lengths'][pred_idx]
        pred = np.argmax(probs[:length], axis=-1)

        # Ensure same length
        min_len = min(len(pred), len(gt))
        pred = pred[:min_len]
        gt = gt[:min_len]

        f1_10_scores.append(f1_score(pred, gt, 0.1))
        f1_25_scores.append(f1_score(pred, gt, 0.25))
        f1_50_scores.append(f1_score(pred, gt, 0.5))

    print(f"F1@10: {np.mean(f1_10_scores):.2f}")
    print(f"F1@25: {np.mean(f1_25_scores):.2f}")
    print(f"F1@50: {np.mean(f1_50_scores):.2f}")

if __name__ == "__main__":
    main()