import pandas as pd
import drms
import os
import numpy as np
import datetime
from astropy.io import fits
from sunpy.net import Fido
from sunpy.net import attrs as a
import time
from utils import ACTIVE_REGIONS_WITH_POSITIVE_FLAREING_EVENTS
import traceback
import multiprocessing
import time

SHARP_SERIES = "hmi.sharp_cea_720s"
SMARP_SERIES = "mdi.smarp_cea_96m"
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


def retrieve_magnetogram(url):
    while True:
        try:
            image = fits.open(url)
            return image

        except Exception as e:
            print(f"Error {e}")
            traceback.print_exc()
            time.sleep(2)


def retrieve_summary_params(client, ar, dataseries):
    while True:
        try:
            keys = client.query(
                f"{dataseries}[{ar}][][]",
                key="T_REC, USFLUXL, MEANGBL, R_VALUE, AREA , HARPNUM, NOAA_ARS",
            )
            return keys
        except Exception as e:
            print(f"Error {e}")
            print(f"{dataseries}[{ar}][][]")
            traceback.print_exc()
            time.sleep(5)


def collect_active_region_summary_parameters(ar_number, dataset):
    client = drms.Client()
    if dataset == "SHARP":
        keys = retrieve_summary_params(client, ar_number, SHARP_SERIES)

        # convert SHARP data cadence to 96m
        # SHARP's original cadence is 720s,by selecting the data with step = 8 we convert it to 96m cadence

        keys = keys.loc[
            [i for j, i in enumerate(keys.index) if j % 8 == 0]
        ].reset_index(drop=True)

    else:
        keys = retrieve_summary_params(client, ar_number, SMARP_SERIES)

    return keys


def collect_magnetograms(client, ar_numbers, dataset):
    for ar_number in ar_numbers:
        if dataset == "SHARP":
            keys, segments = client.query(
                f"{SHARP_SERIES}[{ar_number}][][]", key="T_REC", seg="magnetogram"
            )
            keys = keys.loc[[i for j, i in enumerate(keys.index) if j % 8 == 0]]
            segments = segments.loc[keys.index]

        else:
            keys, segments = client.query(
                f"{SMARP_SERIES}[{ar_number}][][]", key="T_REC", seg="magnetogram"
            )
        download_active_region_magnetograms(
            ar_number, keys.T_REC.tolist(), dataset, segments.magnetogram.tolist()
        )


def download_active_region_magnetograms(ar_number, datetimes, dataset, urls):
    magnetograms = []
    for date_time, url in zip(datetimes, urls):
        full_url = JSOC_URL + url
        print(
            f"downloading magnetogram at datetime {date_time} for {dataset} region {ar_number} from {full_url}"
        )
        if os.path.exists(
            f"data/{dataset}/raw_magnetograms/{ar_number}_{str(parse_tai_string(date_time))}.npy"
        ):
            continue
        image = retrieve_magnetogram(full_url)
        magnetograms.append((url, image[1].data))
        np.save(
            f"data/{dataset}/raw_magnetograms/{ar_number}_{str(parse_tai_string(date_time))}",
            image[1].data,
        )
    return magnetograms


def collect_goes_data():
    event_type = "FL"
    tstart = "2010/01/01"
    tend = "2011/10/29"
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
    filtered_results.write("data/GOES/goes.csv", format="csv", overwrite=True)
    df = pd.read_csv("data/GOES/goes.csv")
    df["event_starttime"] = df["event_starttime"].apply(
        lambda x: datetime.datetime.strptime(x[:-4], "%Y-%m-%d %H:%M:%S")
    )
    df["event_endtime"] = df["event_endtime"].apply(
        lambda x: datetime.datetime.strptime(x[:-4], "%Y-%m-%d %H:%M:%S")
    )
    df.to_csv("data/GOES/goes.csv")


def collect_summary_parameters(ars, dataset="SHARP"):
    client = drms.Client()
    print(f"args: {client}{ars}")
    df = pd.DataFrame(
        columns=[
            "T_REC",
            "MEANGBL",
            "R_VALUE",
            "USFLUXL",
            "AREA",
            "HARPNUM",
            "TARPNUM",
            "NOAA_ARS",
        ]
    )
    for ar in ars:
        print(f" collecting data series for {dataset} region {ar}")
        if os.path.exists(f"data/{dataset}/raw_summary_parameters/{ar}.csv"):
            continue
        ar_data = collect_active_region_summary_parameters(client, ar, dataset=dataset)
        print(ar_data.shape, ar_data.head())
        ar_data["T_REC"] = ar_data["T_REC"].apply(lambda x: parse_tai_string(x))
        ar_data.to_csv(f"data/{dataset}/raw_summary_parameters/{ar}.csv")
        df = pd.concat([df, ar_data], axis=0)

    df = df.sort_values(by=["T_REC"])
    df.to_csv(f"data/{dataset}/summary_paramters_all_active_regions/{dataset}.csv")
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


def collect_summary_parameters_wrapper(args):
    return collect_summary_parameters(client, args)


def square(x):
    return x + x


def get_ars(file):
    lines = open(file).readlines()[1:]
    ars = []
    for line in lines:
        ars.append(line.split()[0])
    return ars


if __name__ == "__main__":
    client = drms.Client()
    harps, tarps = get_ars("data/tarp_harp_to_noaa/harp_noaa.txt"), get_ars(
        "data/tarp_harp_to_noaa/tarp_noaa.txt"
    )
    noaa_harps, noaa_tarps = map_noaa_to_harps_tarps(
        "data/tarp_harp_to_noaa/harp_noaa.txt", "data/tarp_harp_to_noaa/tarp_noaa.txt"
    )
    harp_regions_with_positive_events = set()
    tarp_regions_with_positive_events = set()
    for region in ACTIVE_REGIONS_WITH_POSITIVE_FLAREING_EVENTS:
        if noaa_harps.get(str(region)) != None:
            harp_regions_with_positive_events.add(noaa_harps.get(str(region)))
        if noaa_tarps.get(str(region)) != None:
            tarp_regions_with_positive_events.add(noaa_tarps.get(str(region)))

    # print(noaa_harps.get("12473"), noaa_tarps.get("12473"))
    # print(
    #    len(harp_regions_with_positive_events), len(tarp_regions_with_positive_events)
    #
    # collect_magnetograms(
    #   client, list(harp_regions_with_positive_events)[:2], dataset="SHARP"
    # )
    # collect_summary_parameters(
    #   client, list(harp_regions_with_positive_events)[:2], dataset="SHARP"
    # )
    # collect_magnetograms(
    #     client, set(tarp_regions_with_positive_events[:10]), dataset="SMARP"
    # )

    # collect_summary_parameters(
    #     client, set(tarp_regions_with_positive_events[:10]), dataset="SMARP"
    # )
    # collect_goes_data(start_year, end_year)

    with multiprocessing.Pool(processes=10) as pool:
        args = [(r, "SHARP") for r in list(harp_regions_with_positive_events)[:20]]
        print(args)
        # args = [i for i in range(10)]
        result = pool.starmap(collect_active_region_summary_parameters, args)

    # print(result)
