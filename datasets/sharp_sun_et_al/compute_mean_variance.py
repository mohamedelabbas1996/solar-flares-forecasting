import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import MagnetogramDataset

sharp_df = pd.read_csv("datasets/sharp_sun_et_al/sharp_sun_et_al_filtered.csv")
ds = MagnetogramDataset(sharp_df, magnetograms_dirs=["data/SHARP/sharp_magnetograms_sun_et_al_decompressed/sharp_magnetograms_sun_et_al_compressed_1","data/SHARP/sharp_data_all_magnetograms"])
means = []
vars = []
for idx  in tqdm(range(len(ds))):
    magnetogram = ds[idx][0].numpy()
    mean = np.mean(magnetogram)
    var = np.var(magnetogram)
    means.append(mean)
    vars.append(var)
sharp_df["mean"] = pd.Series(means)
sharp_df["variance"] = pd.Series(vars)
sharp_df.to_csv("datasets/sharp_sun_et_al/sharp_sun_et_al_filtered_mean_variance.csv")    

