import pandas as pd
import datetime
import os
import numpy as np
import torch
import pickle
from utils import get_ars
import os
from collections import defaultdict


# line of sight magnetograms are normalized using the mean and std of SHARP data
# summary parameters are standardized separately, the mean and std for each dataset is used to perform the standardization
def z_score_standardize(x, mean, std):
    epsilon = 0.00001
    return (x - mean) / (std + epsilon)


def preprocess_magnetograms(dataset):
    mean, std = get_magnetograms_mean_std("SHARP")
    raw_magnetograms_path = os.path.join("data", dataset, "raw_magnetograms")
    preprocessed_magnetograms_path = os.path.join(
        "data", dataset, "preprocessed_magnetograms"
    )
    for magnetogram_file in os.listdir(raw_magnetograms_path):
        magnetogram_array = np.load(
            os.path.join(raw_magnetograms_path, magnetogram_file)
        )
        magnetogram_array = z_score_standardize(magnetogram_array, mean, std)
        np.save(
            os.path.join(preprocessed_magnetograms_path, magnetogram_file),
            magnetogram_array,
        )


def preprocess_summary_parameters(dataset):
    raw_summary_parameters_path = os.path.join(
        "data", dataset, "raw_summary_parameters"
    )
    preprocessed_summary_parameters_path = os.path.join(
        "data", dataset, "preprocessed_summary_parameters"
    )
    summary_params_mean_std = get_summary_parameters_mean_std(dataset)
    for ar_file in os.listdir(raw_summary_parameters_path):
        df = read_df_from_csv(os.path.join(raw_summary_parameters_path, ar_file))
        for param in ["USFLUXL", "MEANGBL", "R_VALUE", "AREA"]:
            df[param] = z_score_standardize(
                df[param],
                summary_params_mean_std[param]["mean"],
                summary_params_mean_std[param]["std"],
            )
        write_df_to_csv(df, os.path.join(preprocessed_summary_parameters_path, ar_file))


def get_summary_parameters_mean_std(dataset):
    dataset_path = os.path.join("data", dataset, "raw_summary_parameters")
    params_mean_std = defaultdict(dict)
    all_ars = []
    for ar_file in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, ar_file)):
            continue
        print(f"reading file {os.path.join(dataset_path, ar_file)} ")
        ar_df = pd.read_csv(os.path.join(dataset_path, ar_file))
        all_ars.append(ar_df)
    all_ars_df = pd.concat(all_ars, axis=0, ignore_index=True)
    for param in ["USFLUXL", "MEANGBL", "R_VALUE", "AREA"]:
        params_mean_std[param]["mean"] = all_ars_df[param].mean()
        params_mean_std[param]["std"] = all_ars_df[param].std()
    return params_mean_std


def get_magnetograms_mean_std(dataset):
    n = 0
    s1 = 0
    s2 = 0
    raw_magnetograms_path = os.path.join("data", dataset, "raw_magnetograms")
    for magnetogram_file in os.listdir(raw_magnetograms_path):
        magnetogram_array = np.load(
            os.path.join(raw_magnetograms_path, magnetogram_file)
        )
        magnetogram_array = magnetogram_array[~np.isnan(magnetogram_array)]
        s1 += np.sum(magnetogram_array)
        s2 += np.sum(magnetogram_array**2)
        n += magnetogram_array.size
    mean = s1 / n
    std = np.sqrt(s2 / n - (s1 / n) ** 2)
    return mean, std


# from utils import conver2datetime
def convert2datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")


def get_flare_intensity(flare):
    if flare == "":
        return 0
    intensities = {"D": 0, "A": 10, "B": 20, "C": 30, "M": 40, "X": 50}
    flare_intensity = intensities[flare[0]] + float(flare[1:])
    return flare_intensity


def get_max_flare(flares):
    if not flares:
        return "D0"

    max_flare = "D0"

    for flare in flares:
        if get_flare_intensity(flare) > get_flare_intensity(max_flare):
            max_flare = flare
    return max_flare


def label_data(goes_df, data_series):
    labelled_regions = pd.DataFrame(
        columns=[
            "T_REC",
            "MEANGBL",
            "R_VALUE",
            "AREA",
            "HARPNUM",
            "TARPNUM",
            "NOAA_ARS",
            "label",
            "prediction_period",
            "observation_period",
        ]
    )
    goes_df["event_starttime"] = goes_df["event_starttime"].apply(
        lambda x: convert2datetime(x)
    )
    goes_df["event_endtime"] = goes_df["event_endtime"].apply(
        lambda x: convert2datetime(x)
    )
    for file in os.listdir(f"data/{data_series}"):
        region_df = pd.read_csv(f"data/{data_series}/{file}")
        labelled_region_df = label_region_dataseries(goes_df, region_df)
        labelled_regions = pd.concat(
            [labelled_regions, labelled_region_df], axis=0, ignore_index=True
        )
    labelled_regions.to_csv(f"data/{data_series}/labelled.csv")
    return labelled_regions


def preprocess_region(region_df):
    region_df["magnetogram"] = ""
    region_df["magnetogram_width"] = ""
    region_df["magnetogram_height"] = ""
    region_df["bad_record"] = ""
    region_df["bad_magnetogram"] = ""
    region_df["bad_sample"] = ""
    region_df["region_type"] = ""
    region_df["region_no"] = ""
    region_df["bad_record_reason"] = ""
    region_df["bad_sample_reason"] = ""
    region_df.reset_index(drop=True)
    dataset = "SHARP" if "HARPNUM" in region_df.columns else "SMARP"
    preprocessed_region = pd.DataFrame(
        columns=[
            "T_REC",
            "region_type",
            "region_no",
            "NOAA_ARS",
            "params",
            "magnetogram",
            "label",
        ]
    )
    for idx, row in region_df.iterrows():
        region_df.at[idx, "region_type"] = (
            "harp" if "HARPNUM" in region_df.columns else "tarp"
        )
        region_df.at[idx, "region_no"] = (
            row["HARPNUM"] if "HARPNUM" in region_df.columns else row["TARPNUM"]
        )
        magnetogram_array = np.load(
            f"data/{dataset}/preprocessed_magnetograms/{region_df.at[idx, 'region_no']}_{region_df.at[idx,'T_REC']}.npy"
        )
        magnetogram_height, magnetogram_width = magnetogram_array.shape
        region_df.at[idx, "magnetogram"] = magnetogram_array
        region_df.at[idx, "magnetogram_height"] = magnetogram_height
        region_df.at[idx, "magnetogram_width"] = magnetogram_width

        if (
            region_df.loc[idx, ["MEANGBL", "R_VALUE", "USFLUXL", "AREA"]].isna().sum()
            > 1
        ):
            region_df.at[idx, "bad_record"] = True
            region_df.at[idx, "bad_record_reason"] = (
                region_df.at[idx, "bad_record_reason"] + "," + "nan summary parameter"
            )
        if np.any(np.isnan(region_df.at[idx, "magnetogram"])):
            region_df.at[idx, "bad_magnetogram"] = True
            region_df.at[idx, "bad_record"] = True
            region_df.at[idx, "bad_record_reason"] = (
                region_df.at[idx, "bad_record_reason"] + "," + "magnetogram has nan"
            )
    for idx, row in region_df.iterrows():
        if idx + 15 < region_df.shape[0]:
            sample = region_df.loc[idx : idx + 15, :]
            # print(
            #     region_df.loc[
            #         idx : idx + 15, ["magnetogram_width", "magnetogram_height"]
            #     ]
            # )
            # print(sample["magnetogram_width"])
            sample_magnetogram_width_median = sample["magnetogram_width"].median()
            sample_magnetogram_height_median = sample["magnetogram_height"].median()
            for idx, row in sample.iterrows():
                if (
                    abs(
                        sample.at[idx, "magnetogram_width"]
                        - sample_magnetogram_width_median
                    )
                    > 2
                ):
                    region_df.at[idx, "bad_magnetogram"] = True
                    region_df.at[idx, "bad_record"] = True
                    region_df.at[idx, "bad_record_reason"] = (
                        region_df.at[idx, "bad_record_reason"]
                        + ","
                        + "magnetogram width deviates from the sample width median"
                    )
                if (
                    abs(
                        sample.at[idx, "magnetogram_height"]
                        - sample_magnetogram_height_median
                    )
                    > 2
                ):
                    region_df.at[idx, "bad_magnetogram"] = True
                    region_df.at[idx, "bad_record"] = True
                    region_df.at[idx, "bad_record_reason"] = (
                        region_df.at[idx, "bad_record_reason"]
                        + ","
                        + "magnetogram height deviates from the sample height median"
                    )
            if sample.iloc[-1]["bad_magnetogram"] == True:
                region_df.at[idx, "bad_sample"] = True
                region_df.at[idx, "bad_sample_reason"] = (
                    region_df.at[idx, "bad_sample_reason"] + "," + "bad magnetogram"
                )
            if len(sample[sample["bad_record"] == True]) > 2:
                region_df.at[idx, "bad_sample"] = True
                region_df.at[idx, "bad_sample_reason"] = (
                    region_df.at[idx, "bad_sample_reason"]
                    + ","
                    + "more than 2 bad records"
                )
            # print(f"{idx}", region_df.at[idx, "bad_sample"], region_df.at[idx, "label"])
            if (
                region_df.at[idx, "bad_sample"] != True
                and region_df.at[idx, "label"] != "irrelevant"
            ):
                sample_df = {
                    "T_REC": [region_df.at[idx, "T_REC"]],
                    "region_type": [region_df.at[idx, "region_type"]],
                    "region_no": [region_df.at[idx, "region_no"]],
                    "NOAA_ARS": [region_df.at[idx, "NOAA_ARS"]],
                    "params": [
                        np.array(
                            region_df.loc[
                                idx, ["MEANGBL", "R_VALUE", "USFLUXL", "AREA"]
                            ].tolist()
                        )
                    ],
                    "magnetogram": [sample.iloc[-1]["magnetogram"]],
                    "label": [1 if region_df.at[idx, "label"] == "positive" else 0],
                }
                # print("preprocessed", preprocessed_region.head())
                preprocessed_region = pd.concat(
                    [preprocessed_region, pd.DataFrame.from_dict(sample_df)], axis=0
                )

    # preprocessed_region.to_csv(
    #     f"data/{dataset}/preprocessed_summary_parameters/{region_df.at[0, 'region_no']}.csv"
    # )
    # region_df.to_csv(f"data/{dataset}/{region_df.at[0, 'region_no']}_raw.csv")
    return preprocessed_region


def label_region_summary_parameters(goes_df, sharp_smarp_df):
    noaa_nums = sharp_smarp_df.loc[0, "NOAA_ARS"]
    # print("active reigon number", noaa_nums)
    sharp_smarp_df["label"] = ""
    sharp_smarp_df["observation_period"] = ""
    sharp_smarp_df["prediction_period"] = ""

    # print(goes_df.shape, "shape before filtering")
    # print(goes_df.head())
    goes_df = goes_df[goes_df["ar_noaanum"] == noaa_nums]
    # print(goes_df.shape, "shape after filtering")
    # print(goes_df.head())

    for idx, row in sharp_smarp_df.iterrows():
        # print(type(row["T_REC"]))
        window_start = convert2datetime(row["T_REC"])
        window_end = window_start + datetime.timedelta(hours=24)

        # x1 <= y2 && y1 <= x2
        observation_period_filtered_goes = goes_df[
            (goes_df["event_starttime"] <= window_end)
            & (window_start <= goes_df["event_endtime"])
        ]
        observation_period = observation_period_filtered_goes["fl_goescls"].tolist()

        window_start = window_start + datetime.timedelta(minutes=96)
        window_end = window_start + datetime.timedelta(hours=24)
        prediction_period_filtered_goes = goes_df[
            (goes_df["event_starttime"] <= window_end)
            & (window_start <= goes_df["event_endtime"])
        ]
        prediction_period = prediction_period_filtered_goes["fl_goescls"].tolist()
        # print(observation_period, get_max_flare(observation_period))
        # print(prediction_period, get_max_flare(prediction_period))
        # print(prediction_period_filtered_goes, window_start, window_end)
        # print(observation_period_filtered_goes, window_start, window_end)
        max_obserevation = get_max_flare((observation_period))
        max_prediction = get_max_flare((prediction_period))
        # if we have strong event in the prediction period
        label = ""
        if get_flare_intensity(max_prediction) >= get_flare_intensity("M1.0"):
            label = "positive"
        # both prediction period and observation period are quiet
        elif get_flare_intensity(max_obserevation) == get_flare_intensity(
            "D0"
        ) and get_flare_intensity(max_prediction) == get_flare_intensity("D0"):
            label = "negative"
        else:
            label = "irrelevant"

        sharp_smarp_df.loc[idx, "observation_period"] = max_obserevation
        sharp_smarp_df.loc[idx, "prediction_period"] = max_prediction
        sharp_smarp_df.loc[idx, "label"] = label

        # print(filtered_goes)
    # print(sharp_smarp_df[sharp_smarp_df["label"] == "negative"])
    # print(sharp_smarp_df[["observation_period", "prediction_period", "label"]])
    sharp_smarp_df.to_csv("sharp_smarp_labelled.csv")
    return sharp_smarp_df


def read_df_from_csv(path):
    return pd.read_csv(path)


def write_df_to_csv(df, path):
    df.to_csv(path)


def read_df_from_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def write_df_to_pickle(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)


def get_all_regions_ar_params_magnetograms(dataset):
    summary_params_path = f"data/{dataset}/summary_parameters_magnetograms"
    pkl_files = [file for file in os.listdir(summary_params_path)]
    all_preprocessed_active_regions = []

    for file in pkl_files:
        df = read_df_from_pickle(os.path.join(summary_params_path, file))
        all_preprocessed_active_regions.append(df)

    all_preprocessed_active_regions = pd.concat(
        all_preprocessed_active_regions, axis=0, ignore_index=True
    )
    return all_preprocessed_active_regions


def preprocess_goes(path, file_name):
    goes_df = read_df_from_csv(os.path.join(path, file_name))
    goes_df["event_starttime"] = goes_df["event_starttime"].apply(
        lambda x: convert2datetime(x)
    )
    goes_df["event_endtime"] = goes_df["event_endtime"].apply(
        lambda x: convert2datetime(x)
    )
    write_df_to_pickle(goes_df, os.path.join(path, "goes.pkl"))


if __name__ == "__main__":
    print("Preprocessing GOES data ...")
    preprocess_goes("data/GOES", "goes.csv")
    goes_df = read_df_from_pickle("data/GOES/goes.pkl")

    print("Reading HARPS and TARPs ...")
    harps, tarps = get_ars("data/tarp_harp_to_noaa/harp_noaa.txt"), get_ars(
        "data/tarp_harp_to_noaa/tarp_noaa.txt"
    )
    print(
        f"there are {len(harps)} sharp active regions and {len(tarps)} smarps active regions"
    )
    harps = [(region_no, "SHARP") for region_no in harps[:5]]
    tarps = [(region_no, "SMARP") for region_no in tarps[:5]]
    harps_tarps = harps + tarps

    print("Preprocessing magnetograms")
    preprocess_magnetograms("SHARP")
    preprocess_magnetograms("SMARP")

    print("Preprocessing summary parameters ...")
    preprocess_summary_parameters("SHARP")
    preprocess_summary_parameters("SMARP")

    for region, dataset in harps_tarps:
        print(f"preprocessing {dataset} region {region}")
        ar_df = read_df_from_csv(
            f"data/{dataset}/preprocessed_summary_parameters/{region}.csv"
        )
        labelled_region = label_region_summary_parameters(goes_df, ar_df)
        print(labelled_region.head())
        preprocessed_region = preprocess_region(labelled_region)
        write_df_to_pickle(
            preprocessed_region,
            f"data/{dataset}/summary_parameters_magnetograms/{region}.pkl",
        )
        print(preprocessed_region.head())

    for dataset in ["SHARP", "SMARP"]:
        all_ars_params_df = get_all_regions_ar_params_magnetograms(dataset)
        write_df_to_pickle(
            all_ars_params_df,
            f"data/{dataset}/{dataset}.pkl",
        )
