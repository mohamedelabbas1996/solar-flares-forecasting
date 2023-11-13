python -m venv solar-flares-forecasting
source solar-flares-forecasting/bin/activate
pip install -r requirements.txt

mkdir data
mkdir data/SHARP
mkdir data/SHARP/raw_summary_parameters
mkdir data/SHARP/preprocessed_summary_parameters
mkdir data/SHARP/summary_parameters_magnetograms
mkdir data/SHARP/raw_magnetograms
mkdir data/SHARP/preprocessed_magnetograms
mkdir data/SHARP/summary_paramters_all_active_regions

mkdir data/SMARP
mkdir data/SMARP/raw_summary_parameters
mkdir data/SMARP/preprocessed_summary_parameters
mkdir data/SMARP/summary_parameters_magnetograms
mkdir data/SMARP/raw_magnetograms
mkdir data/SMARP/preprocessed_magnetograms
mkdir data/SMARP/summary_paramters_all_active_regions

mkdir data/GOES
wget https://raw.githubusercontent.com/mohamedelabbas1996/solar-flares-forecasting/main/data/GOES/goes.csv -O data/GOES/goes.csv

mkdir data/tarp_harp_to_noaa
wget http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt -O data/tarp_harp_to_noaa/harp_noaa.txt
wget http://jsoc.stanford.edu/doc/data/mdi/all_tarps_with_noaa_ars.txt -O data/tarp_harp_to_noaa/tarp_noaa.txt


echo "Downloading data"
# python datasets/collect_data.py SHARP
# python datasets/collect_data.py SMARP

echo 