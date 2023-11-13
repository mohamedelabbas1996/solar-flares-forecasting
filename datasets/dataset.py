from torch.utils.data.dataset import Dataset
from typing import Any
from datetime import datetime
import drms
import pandas as pd
import pickle
import torch
import torchvision.transforms as T


class SHARPDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        return 0


class SolarFlaresData(Dataset):
    def __init__(self, df):
        self.df = df
        self.resize_transform = T.Resize((128, 128))

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.df.loc[idx, "params"]),
            self.resize_transform(
                torch.from_numpy(self.df.loc[idx, "magnetogram"]).unsqueeze(0)
            ),
            torch.tensor(self.df.loc[idx, "label"]),
        )

    def __len__(self):
        return self.df.shape[0]


class SMARPDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        return 0


if __name__ == "__main__":
    with open(
        "/Users/mohamedelabbas/solar-flares-forecasting/solar-flares-forecasting/data/sharp/preprocessed/all_sharps.pkl",
        "rb",
    ) as f:
        df = pickle.load(f)
    sf = SolarFlaresData(df)
    print(sf[10][0].shape, sf[10][1].shape, sf[10][2])
    print(len(sf))
