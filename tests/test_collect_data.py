from datasets.collect_data import (
    collect_dataseries,
    collect_active_region_magnetograms,
    collect_active_region_data_series,
    collect_goes_data,
)


def test_collect_goes_data():
    collect_goes_data()
