import datetime
import pandas as pd
import pickle


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


def map_noaa_to_harps_tarps(harps_noaa, tarps_noaa):
    df_harps = pd.read_csv(harps_noaa, sep=" ")
    df_tarps = pd.read_csv(tarps_noaa, sep=" ")
    noaa_tarps = {}
    noaa_harps = {}
    for idx, row in df_harps.iterrows():
        harp_num = row["HARPNUM"]
        noaa_num = row["NOAA_ARS"]
        for ar in noaa_num.split(","):
            noaa_harps[ar] = harp_num
    for idx, row in df_tarps.iterrows():
        tarp_num = row["TARPNUM"]
        noaa_num = row["NOAA_ARS"]
        for ar in noaa_num.split(","):
            noaa_tarps[ar] = harp_num
    return noaa_harps, noaa_tarps


def get_ars(file):
    lines = open(file).readlines()[1:]
    ars = []
    for line in lines:
        ars.append(line.split()[0])
    return ars


def convert2datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str[:-4], "%Y-%m-%d %H:%M:%S")


def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return datetime.datetime(year, month, day, hour, minute)
