from src.data.datasets.ddtr_logs import DDTRLogDataset
from src.data.datasets.fall_seg import FallSegDataset
from src.data.datasets.fall_detection import FallDetectionDataset
from src.data.datasets.gtea_hf import GTEAHFFeatureDataset

DATASET_REGISTRY = {
    "ddtr_logs": DDTRLogDataset,
    "fall_seg": FallSegDataset,
    "fall_detection": FallDetectionDataset,
    "gtea_hf": GTEAHFFeatureDataset,
}


def get_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "DDTRLogDataset",
    "FallSegDataset",
    "FallDetectionDataset",
    "GTEAHFFeatureDataset",
    "get_dataset",
]
