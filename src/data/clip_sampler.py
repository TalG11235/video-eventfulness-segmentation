import numpy as np


def sample_clip(features, labels, clip_len, random_start, pad_value, rng):
    length = min(len(features), len(labels))
    features = features[:length]
    labels = labels[:length]

    if length >= clip_len:
        start = rng.randint(0, length - clip_len) if random_start else 0
        end = start + clip_len
        feat = features[start:end]
        lab = labels[start:end]
        mask = np.ones((clip_len,), dtype=bool)
    else:
        start = 0
        pad = clip_len - length
        feat = np.concatenate(
            [features, np.zeros((pad, features.shape[1]), dtype=features.dtype)], axis=0
        )
        lab = np.concatenate(
            [labels, np.full((pad,), pad_value, dtype=labels.dtype)], axis=0
        )
        mask = np.concatenate(
            [np.ones((length,), dtype=bool), np.zeros((pad,), dtype=bool)], axis=0
        )

    return feat, lab, mask, start, length
