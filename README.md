# Umbra
This repository contains artifacts for the accepted Mobicom'23 paper "Transmitting, Fast and Slow: Scheduling Satellite Traffic through Space and Time".

## Setting up the environment
We provide two options to set up the environment: using Docker or installing the dependencies manually.

### Using Docker
We provide a Dockerfile that can be used to build a Docker image with all the dependencies installed. To build the image, run the following command:
```
docker build -t umbra .
```
To run the image, use the following command:
``` 
docker run -it umbra
```

### Installing dependencies manually
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

If you don't have an API key, you can download the metadata gathered by us from [XXX](XXX). We provide the metadata for the first 4 days of June, July and August of 2021, which is the data we used in our experiments.

_Disclaimer: The metadata is downloaded from the Planet website and is subject to their terms and conditions._ 

_Planet has made a re-arrangement of its image assets in March 2022 and [deprecated](https://developers.planet.com/tag/deprecating.html#deprecating-psscene3band-and-psscene4band) the metadata we used for experiments in the paper. The downloading script we provide uses the new scheme, so that you can download other dates to explore. However, the metadata downloaded using the script will be different from the one used in the experiment. If you want to replicate our experiment results, please use the metadata file we provide._
