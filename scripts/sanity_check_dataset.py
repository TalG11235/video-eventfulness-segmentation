from torch.utils.data import DataLoader

from src.data import get_dataset


def main():
    ds = get_dataset(
        "ddtr_logs",
        manifest_path="data/ddtr/train.jsonl",
        clip_len=32,
        random_start=True,
        input_type="frames",
        seed=0,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    print(batch["inputs"].shape)  # [B,T,C,H,W] or [B,T,D]
    print(batch["labels"].shape)  # [B,T]
    print(batch["mask"].shape)    # [B,T]
    print(batch["labels"][0][:10], batch["mask"][0][:10])


if __name__ == "__main__":
    main()
