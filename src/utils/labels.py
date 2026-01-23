import numpy as np

IGNORE_INDEX = -100

def segments_to_frame_labels(num_frames: int, segments, bg: int = 0, fg: int = 1):
    """
    segments: list of (start_frame, end_frame_exclusive) that are "event"
    returns labels shape [num_frames]
    """
    y = np.full((num_frames,), bg, dtype=np.int64)
    for s, e in segments:
        s = max(0, int(s))
        e = min(num_frames, int(e))
        if e > s:
            y[s:e] = fg
    return y

def pad_to_T(arr, T: int, pad_value):
    """
    Pads 1D or 4D arrays/tensors to length T on first dim.
    """
    L = arr.shape[0]
    if L >= T:
        return arr[:T]
    pad_shape = (T - L,) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)
