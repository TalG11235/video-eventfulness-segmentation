from torch.utils.data import DataLoader
# from src.data.datasets.my_dataset import MyDataset

def main():
    ds = MyDataset(T=32, image_size=224, stride=1, random_start=True)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    print(batch["frames"].shape)  # [B,T,3,H,W]
    print(batch["labels"].shape)  # [B,T]
    print(batch["mask"].shape)    # [B,T]
    print(batch["labels"][0][:10], batch["mask"][0][:10])

if __name__ == "__main__":
    main()
