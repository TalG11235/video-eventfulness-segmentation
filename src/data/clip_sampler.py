from dataclasses import dataclass
import random

@dataclass
class ClipSpec:
    start: int
    end: int  # exclusive

class FixedLengthClipSampler:
    """
    Samples clips of fixed length T from a video of length L frames.
    If L < T, we still return a clip spec and padding will handle the rest.
    """
    def __init__(self, T: int, stride: int = 1, random_start: bool = True):
        self.T = T
        self.stride = stride
        self.random_start = random_start

    def sample(self, L: int) -> ClipSpec:
        # Effective clip span in original frame indices
        span = (self.T - 1) * self.stride + 1
        if L <= 0:
            return ClipSpec(0, 0)

        if L <= span:
            return ClipSpec(0, L)

        max_start = L - span
        start = random.randint(0, max_start) if self.random_start else 0
        end = start + span
        return ClipSpec(start, end)
