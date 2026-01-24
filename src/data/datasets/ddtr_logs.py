from src.data.base import ManifestVideoDataset, IGNORE_INDEX


class DDTRLogDataset(ManifestVideoDataset):
    def __init__(
        self,
        manifest_path: str,
        clip_len: int,
        random_start: bool,
        input_type: str,
        seed: int = 0,
    ):
        super().__init__(
            manifest_path=manifest_path,
            clip_len=clip_len,
            random_start=random_start,
            input_type=input_type,
            pad_value=IGNORE_INDEX,
            seed=seed,
        )
