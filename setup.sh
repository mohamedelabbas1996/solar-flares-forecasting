mkdir data
echo "Downloading csv files ..."
mkdir data/SHARP
gdown 1j9evvHkXrc-e-hgdw7LuOS_Mt7QSMP1v -O data/SHARP

echo "Downloading magnetograms ..."
mkdir data/SHARP/magnetograms
gdown 1SvKi9Ife0Q5K2Bro0K3vtG0xGWcWZ3S0 -O data/SHARP/magnetograms

mkdir checkpoints
# mkdir data/SHARP/raw_summary_parameters
# mkdir data/SHARP/preprocessed_summary_parameters
# mkdir data/SHARP/summary_parameters_magnetograms
# mkdir data/SHARP/raw_magnetograms
# mkdir data/SHARP/preprocessed_magnetograms
# mkdir data/SHARP/summary_paramters_all_active_regions

# mkdir data/SMARP
# mkdir data/SMARP/raw_summary_parameters
# mkdir data/SMARP/preprocessed_summary_parameters
# mkdir data/SMARP/summary_parameters_magnetograms
# mkdir data/SMARP/raw_magnetograms
# mkdir data/SMARP/preprocessed_magnetograms
# mkdir data/SMARP/summary_paramters_all_active_regions

mkdir data/GOES
wget https://raw.githubusercontent.com/mohamedelabbas1996/solar-flares-forecasting/main/data/GOES/goes.csv -O data/GOES/goes.csv
gdown 1xXS89g1DeYTu8aFZTez1BUp0NgOqzium -O data/GOES/
mkdir data/tarp_harp_to_noaa
wget http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt -O data/tarp_harp_to_noaa/harp_noaa.txt
wget http://jsoc.stanford.edu/doc/data/mdi/all_tarps_with_noaa_ars.txt -O data/tarp_harp_to_noaa/tarp_noaa.txt


