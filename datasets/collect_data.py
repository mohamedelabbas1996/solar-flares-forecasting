import pandas as pd
import drms
import os
from astropy.io import fits
from sunpy.net import Fido
from sunpy.net import attrs as a


def collect_active_region_data_series(client, ar_number, dataseries="sharp"):
    if dataseries == "sharp":
        keys = client.query(
            f"hmi.sharp_cea_720s[{ar_number}][][]",
            key="T_REC, MEANGBL, R_VALUE, AREA , HARPNUM, NOAA_ARS",
        )

    else:
        keys = client.query(
            f"mdi.smarp_cea_720s[{ar_number}][][]",
            key="T_REC, MEANGBL, R_VALUE, AREA, TARPNUM, NOAA_ARS",
        )

    return keys


def collect_active_region_magnetograms(client, ar_numbers, dataseries="sharp"):
    pass


def collect_goes_data():
    event_type = "FL"
    tstart = "2010/10/28"
    tend = "2013/10/29"
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

    print(filtered_results)
    filtered_results.write("data/goes.csv", format="csv", overwrite=True)


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

        df = pd.concat([df, ar_data], axis=0)

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
    collect_dataseries(client, harps[:20], "sharp")
    collect_dataseries(client, tarps[:20], "smarp")
    collect_goes_data()
