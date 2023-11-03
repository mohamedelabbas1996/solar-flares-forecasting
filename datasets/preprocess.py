import pandas as pd
import datetime


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
        return ""

    max_flare = "D0"

    for flare in flares:
        if get_flare_intensity(flare) > get_flare_intensity(max_flare):
            max_flare = flare
    return max_flare


def label_data(goes_data, sharp_smarp_data):
    goes_df: pd.DataFrame = pd.read_csv(goes_data)
    sharp_smarp_df: pd.DataFrame = pd.read_csv(sharp_smarp_data)

    sharp_smarp_df["label"] = ""
    sharp_smarp_df["observation_period"] = ""
    sharp_smarp_df["prediction_period"] = ""
    goes_df["event_starttime"] = goes_df["event_starttime"].apply(
        lambda x: convert2datetime(x)
    )
    goes_df["event_endtime"] = goes_df["event_endtime"].apply(
        lambda x: convert2datetime(x)
    )
    for idx, row in sharp_smarp_df.iterrows():
        print(type(row["T_REC"]))
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
        print(observation_period, get_max_flare(observation_period))
        print(prediction_period, get_max_flare(prediction_period))
        print(prediction_period_filtered_goes, window_start, window_end)
        print(observation_period_filtered_goes, window_start, window_end)
        max_obserevation = get_max_flare((observation_period))
        max_prediction = get_max_flare((prediction_period))
        # if we have strong event in the prediction period
        label = ""
        if get_flare_intensity(max_prediction) >= get_flare_intensity("M1.0"):
            label = "positive"
        # both prediction period and observation period are quiet
        elif (
            get_flare_intensity(max_obserevation) == get_flare_intensity("D0")
            and get_flare_intensity(max_prediction) == "D0"
        ):
            label = "negative"
        else:
            label = "irrelevant"

        sharp_smarp_df.loc[idx, "observation_period"] = max_obserevation
        sharp_smarp_df.loc[idx, "prediction_period"] = max_prediction
        sharp_smarp_df.loc[idx, "label"] = label

        # print(filtered_goes)
    print(sharp_smarp_df[sharp_smarp_df["label"] == "negative"])
    print(sharp_smarp_df[["observation_period", "prediction_period", "label"]])
    sharp_smarp_df.to_csv("sharp_smarp_labelled.csv")


def preprocess_data_series():
    pass


def preprocess_magnetograms():
    pass


if __name__ == "__main__":
    label_data("data/goes.csv", "data/sharp.csv")
