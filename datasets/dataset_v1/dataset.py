from torch.utils.data.dataset import Dataset
from typing import Any
from datetime import datetime
import drms
import pandas as pd
import pickle
import torch
import numpy as np
import torchvision.transforms as T
import torch
from icecream import ic


def read_df_from_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


class SolarFlaresData(Dataset):
    def __init__(self, df, random_undersample=True):
        self.resize_transform = T.Resize((128, 128))
        self.df = self.random_undersample(df) if random_undersample else df

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.df.loc[idx, "params"]).to(torch.float32),
            self.resize_transform(
                torch.from_numpy(self.df.loc[idx, "magnetogram"])
                .unsqueeze(0)
                .to(torch.float32)
            ),
            torch.tensor(self.df.loc[idx, "label"], dtype=torch.float32).reshape(
                1,
            ),
        )

    def random_undersample(self, df):
        positive_samples_count = len(df[df.label == 1])
        negative_samples_count = len(df[df.label == 0])
        samples_dropped_count = negative_samples_count - positive_samples_count
        indices_to_drop = np.random.choice(
            df[df.label == 0].index, samples_dropped_count, replace=False
        )
        df_dropped = df.drop(indices_to_drop)
        return df_dropped.reset_index()

    def __len__(self):
        return self.df.shape[0]
class MagnetogramDataset(Dataset):
    def __init__(self, dataframe, magnetograms_dir="data/SHARP/magnetograms"):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the paths to magnetograms and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.overall_mean = dataframe['mean'].mean()
    
        # Compute the average of the variances from the 'variance' column
        V_avg = dataframe['variance'].mean()
        
        # Compute the variance among the means and scale by the number of observations
        V_means = dataframe['mean'].var()
        
        # Combine V_avg and V_means to get the overall variance
        self.overall_variance = V_avg + V_means
    
        self.magnetograms_dir = magnetograms_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        magnetogram_path = self.dataframe.iloc[idx]['magnetogram']
        magnetogram_region_no = self.dataframe.iloc[idx]['harp_no']
        label = self.dataframe.iloc[idx]['label']
        
        # Load the magnetogram; assuming it's stored as a NumPy array
        magnetogram = np.load(self.magnetograms_dir+f"/{magnetogram_region_no}/"+magnetogram_path[1:].replace("/","_")+".npy")
        
        # Convert magnetogram and label to PyTorch tensors
        magnetogram = torch.from_numpy(magnetogram).float()
        overall_std = np.sqrt(self.overall_variance)
    
        # Normalize the magnetogram
        normalized_magnetogram = (magnetogram - self.overall_mean) / overall_std
        label = torch.tensor(1).long()  if label=="positive"  else  torch.tensor(0).long()# Assuming label is an integer class label
        
        
        return normalized_magnetogram, label


if __name__ == "__main__":
    df = read_df_from_pickle("data/SHARP/SHARP.pkl")

    sf = SolarFlaresData(df)
    ic(sf[10][0], sf[10][1], sf[10][2].shape)
    print(len(sf))
