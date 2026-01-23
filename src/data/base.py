import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.video_reader import OpenCVVideoReader
from src.data.clip_sampler import FixedLengthClipSampler
from src.utils.labels import IGNORE_INDEX

class BaseBinaryEventDataset(Dataset):
    """
    Subclass must implement:
      - self.items: list of dicts with keys: video_path, segments (list), video_id
      - get_num_frames(video_path) (or store it in items)
    """
    def __init__(self, T: int, image_size: int = 224, stride: int = 1, random_start: bool = True):
        self.T = T
        self.sampler = FixedLengthClipSampler(T=T, stride=stride, random_start=random_start)

        self.tf = transforms.Compose([
            transforms.ToTensor(),  # [C,H,W] in 0..1
            transforms.Resize((image_size, image_size)),
            # optionally add normalization later (ImageNet)
        ])

    def __len__(self):
        return len(self.items)

    def _load_video_len_and_fps(self, video_path: str):
        vr = OpenCVVideoReader(video_path)
        L, fps = vr.length, vr.fps
        vr.close()
        return L, fps

    def _segments_to_labels(self, L: int, segments):
        y = np.zeros((L,), dtype=np.int64)
        for s, e in segments:
            s = max(0, int(s)); e = min(L, int(e))
            if e > s:
                y[s:e] = 1
        return y

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = item["video_path"]
        segments = item.get("segments", [])
        video_id = item.get("video_id", str(idx))

        L, fps = self._load_video_len_and_fps(path)
        clip = self.sampler.sample(L)

        # Build indices for frames we actually fetch (respect stride)
        indices = list(range(clip.start, clip.end, self.sampler.stride))
        # indices length <= T, will pad later
        vr = OpenCVVideoReader(path)
        raw_frames = vr.get_frames(indices)
        vr.close()

        # Convert to tensors + pad missing frames
        frames = []
        valid = []
        for f in raw_frames:
            if f is None:
                valid.append(False)
                frames.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            else:
                valid.append(True)
                frames.append(self.tf(f))

        frames = torch.stack(frames, dim=0)  # [Lclip,3,H,W]
        mask = torch.tensor(valid, dtype=torch.bool)  # [Lclip]

        # Build per-frame labels for the whole video, then slice to clip indices
        full_y = self._segments_to_labels(L, segments)
        y = torch.tensor([full_y[i] if (0 <= i < L) else 0 for i in indices], dtype=torch.long)

        # Pad to T
        if frames.shape[0] < self.T:
            pad_n = self.T - frames.shape[0]
            frames = torch.cat([frames, torch.zeros((pad_n, *frames.shape[1:]), dtype=frames.dtype)], dim=0)
            y = torch.cat([y, torch.full((pad_n,), IGNORE_INDEX, dtype=y.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros((pad_n,), dtype=torch.bool)], dim=0)
        else:
            frames = frames[:self.T]
            y = y[:self.T]
            mask = mask[:self.T]

        return {
            "frames": frames,        # [T,3,H,W]
            "labels": y,             # [T]
            "mask": mask,            # [T]
            "meta": {
                "video_id": video_id,
                "start_frame": clip.start,
                "fps": fps,
                "orig_len": L,
            }
        }
