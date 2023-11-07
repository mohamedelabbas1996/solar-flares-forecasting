import pandas as pd
import drms
import os
import numpy as np
import datetime
from astropy.io import fits
from sunpy.net import Fido
from sunpy.net import attrs as a


SHARP_SERIES = "hmi.sharp_cea_720s"
SMARP_SERIES = "mdi.smarp_cea_720s"
JSOC_URL = "http://jsoc.stanford.edu"


def convert2datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str[:-4], "%Y-%m-%d %H:%M:%S")


def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return datetime.datetime(year, month, day, hour, minute)


def collect_active_region_data_series(client, ar_number, dataseries="sharp"):
    if dataseries == "sharp":
        keys = client.query(
            f"{SHARP_SERIES}[{ar_number}][][]",
            key="T_REC, MEANGBL, R_VALUE, AREA , HARPNUM, NOAA_ARS",
        )

    else:
        keys = client.query(
            f"{SMARP_SERIES}[{ar_number}][][]",
            key="T_REC, MEANGBL, R_VALUE, AREA, TARPNUM, NOAA_ARS",
        )

    return keys


def collect_active_region_magnetograms(client, ar_numbers, dataseries="sharp"):
    for ar_number in ar_numbers:
        if dataseries == "sharp":
            keys, segments = client.query(
                f"{SHARP_SERIES}[{ar_number}][][]", key="T_REC", seg="magnetogram"
            )

        else:
            keys, segments = client.query(
                f"{SMARP_SERIES}[{ar_number}][][]", seg="magnetogram"
            )
        return download_magnetograms(
            ar_numbers, keys.T_REC.tolist(), dataseries, segments.magnetogram.tolist()
        )


def download_magnetograms(ar_number, datetimes, series, urls):
    magnetograms = []
    for date_time, url in zip(datetimes, urls):
        full_url = JSOC_URL + url
        print(
            f"downloading magnetogram for {series} region {ar_number} from {full_url}"
        )
        image = fits.open(full_url)
        magnetograms.append((url, image[1].data))
        np.save(f"data/magnetograms/{ar_number}_{series}_{date_time}", image[1].data)
    return magnetograms
    

def collect_goes_data():
    event_type = "FL"
    tstart = "1996/01/01"
    tend = "2020/10/29"
    print(f"collecting goes data from {tstart} to {tend}")
    result = Fido.search(
        a.Time(tstart, tend),
        a.hek.EventType(event_type),
        a.hek.FL.GOESCls <= "X9.9",
        a.hek.OBS.Observatory == "GOES",
    )

    hek_results = result["hek"]
    filtered_results = hek_results[
        "event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum"
    ]

    print((filtered_results["event_starttime"]))
    filtered_results.write("data/goes.csv", format="csv", overwrite=True)
    df = pd.read_csv("data/goes.csv")
    df["event_starttime"] = df["event_starttime"].apply(
        lambda x: datetime.datetime.strptime(x[:-4], "%Y-%m-%d %H:%M:%S")
    )
    df["event_endtime"] = df["event_endtime"].apply(
        lambda x: datetime.datetime.strptime(x[:-4], "%Y-%m-%d %H:%M:%S")
    )
    df.to_csv("data/goes.csv")


def collect_dataseries(client, ars, dataseries="sharp"):
    df = pd.DataFrame(
        columns=[
            "T_REC",
            "MEANGBL",
            "R_VALUE",
            "AREA",
            "HARPNUM",
            "TARPNUM",
            "NOAA_ARS",
        ]
    )
    for ar in ars:
        print(f" collecting data series for {dataseries} region {ar}")
        ar_data = collect_active_region_data_series(client, ar)
        ar_data["T_REC"] = ar_data["T_REC"].apply(lambda x: parse_tai_string(x))
        ar_data.to_csv(f"data/{dataseries}/{ar}.csv")
        df = pd.concat([df, ar_data], axis=0)
    df = df.sort_values(by=["T_REC"])
    df.to_csv(f"data/{dataseries}.csv")
    return df


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


if __name__ == "__main__":
    client = drms.Client()
    harps, tarps = get_ars("datasets/harp_noaa.txt"), get_ars("datasets/tarp_noaa.txt")
    # collect_active_region_magnetograms(client, harps[:1])
    #collect_dataseries(client, harps[:1], "sharp")
    #collect_dataseries(client, tarps[:1], "smarp")
    collect_goes_data()
    # image = collect_active_region_magnetograms(client, 2)
    # print(np.isnan(image).any(axis=0))
    # print(image.shape, image)
