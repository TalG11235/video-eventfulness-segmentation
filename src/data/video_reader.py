import cv2
import numpy as np

class OpenCVVideoReader:
    """
    Minimal reader: random access by frame index (not super fast but simple).
    Consider swapping to decord later if needed.
    """
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def get_frames(self, indices):
        frames = []
        for idx in indices:
            if idx < 0 or idx >= self.length:
                frames.append(None)
                continue
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = self.cap.read()
            if not ok:
                frames.append(None)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        return frames

    def close(self):
        self.cap.release()
