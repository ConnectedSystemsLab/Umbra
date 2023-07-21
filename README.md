# Umbra
This repository contains artifacts for the accepted Mobicom'23 paper "Transmitting, Fast and Slow: Scheduling Satellite Traffic through Space and Time".

## Setting up the environment
We provide the environment using `conda`.

To install the dependencies manually, run the following command:
```
conda env create -f umbra.yml
```
This will create a conda environment named `umbra` with all the dependencies installed. To activate the environment, run the following command:
```
conda activate umbra
```

## Running the experiments
### Step 1: Downloading the Planet metadata
To download the Planet metadata, you have to first create an account on the Planet website. Then, you can download the metadata using the following command:
```
export PLANET_API_KEY=<your_api_key>
python preprocessing/download_planet_metadata.py --geojson_file data/california.geojson --output_pickle_file <output_file_name> \
--year <YEAR> --month <MONTH> --days <Number of days you want to download>
```
This will download the metadata for the Planet images that we used in our experiments and store it in the path you specified as the output file name.

If you don't have an API key, you can download the metadata gathered by us from [this link](https://uofi.box.com/v/umbra-planet-metadata). We provide the metadata for the first 4 days of June, July and August of 2021, which is the data we used in our experiments.

_Disclaimer: The metadata is downloaded from the Planet website and is subject to their terms and conditions._ 

_Planet has made a re-arrangement of its image assets in March 2022 and [deprecated](https://developers.planet.com/tag/deprecating.html#deprecating-psscene3band-and-psscene4band) the metadata we used for experiments in the paper. The downloading script we provide uses the new scheme, so that you can download other dates to explore. However, the metadata downloaded using the script will be different from the one used in the experiment. If you want to replicate our experiment results, please use the metadata file we provide._

### Step 2: Preprocessing the data
During preprocessing, you need to do 2 things (in order):
- Convert the Planet metadata to the format that can be used by the simulator
- Pre-calculate the satellite visibility and bandwidth traces

To convert the metadata, run the following command:
```
export PYTHONPATH=<path_to_umbra_repo>
python preprocessing/convert_planet_metadata.py --input_pickle_file <metadata_file> --output_pickle_file <mapping_file>
```
The pre-processed metadata will be stored in the path you specified as the output file name. This file will be used in the next step. You can also download the pre-processed metadata from [this link](https://uofi.box.com/v/umbra-mapping).

To pre-calculate the satellite visibility and bandwidth traces, you would need a DarkSky API key to get the weather information which is required for calculating the bandwidth. After getting your API key, run the following command:
```
export PYTHONPATH=<path_to_umbra_repo>
export DARKSKY_KEY=<your_api_key>
python preprocessing/calculate_bandwidth.py --gs_config <gs_config_file> --satellite_mapping_file <satellite_mapping_file> --start_time YYYY-MM-DDTHH:MM:SS --end_time YYYY-MM-DDTHH:MM:SS --output_file <output_file> 
```

Please make sure to calculate at least a 40-day-long trace for each month.

_Note: We used DarkSky API to get the weather data. However, the DarkSky API is deprecated and is no longer available. You can run the calculation script which will assume clear weather for all the days since it cannot obtain weather info._