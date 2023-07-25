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

## Reproducing the results
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

The `export` command is to add the path to the `umbra` repository to the `PYTHONPATH` environment variable. This is required so that the script can import the dependencies within the `umbra` repository. Make sure to have this environment variable set before running any python script in this repository.

The pre-processed metadata will be stored in the path you specified as the output file name. This file will be used in the next step. You can also download the pre-processed metadata from [this link](https://uofi.box.com/v/umbra-mapping).

To pre-calculate the satellite visibility and bandwidth traces, you would need a DarkSky API key to get the weather information which is required for calculating the bandwidth. After getting your API key, run the following command:
```
export DARKSKY_KEY=<your_api_key>
python preprocessing/calculate_bandwidth.py --gs_config <gs_config_file> --satellite_mapping_file <satellite_mapping_file> --start_time YYYY-MM-DDTHH:MM:SS --end_time YYYY-MM-DDTHH:MM:SS --output_file <output_file> 
```

Please make sure to calculate at least a 40-day-long trace for each month.

_Note: We used DarkSky API to get the weather data. However, the DarkSky API is [deprecated](https://blog.darksky.net/) and is no longer available. You can run the calculation script which will assume clear weather for all the days since it cannot obtain weather info._

Then pre-computed bandwidth files are available [here](https://uofi.box.com/v/umbra-bandwidth).

### Step 3: Running the experiments
The next step is to run the main experiments and generate the raw results. Our main experiments can be devided into several parts
#### Main Experiment
The main experiment produces results in Table 2, Figure 11 and 12. In the experiment, you need to run `Umbra` together with all the baselines.

To run Umbra, run the following command:
```
python experiments/binary_search.py --image_mapping_info <mapping_file> --gs_config <ground_station_config> --start_time YYYY-MM-DDTHH:MM:SS --cache_file <bandwidth_file> --result_dir <result_directory>
```

Make sure that the `start_time` matches the `mapping_file` and `bandwidth_file` matches the `ground_station_config` and `start_time`. Some raw simulation results will be stored in the `result_directory` you specified. Each simulation instance will generate a folder containing multiple files as the results. These files will be used to generate the plos in the paper.

You can find the ground station config files we used in the paper in the `config` folder of this repository. 

To run the baselines, use the files `experiments/basic_heuristic.py`, `experiments/smart_heristic.py` and the same arguments as the `Umbra` command. `binary_search.py` generates results for both `Umbra` and the `greedy` baseline, and `basic_heuristic.py` generates results for the `naive` baseline. `smart_heuristic.py` generates results for the `smart` baseline.

#### Heterogeneous Backhaul
For heterogeneous backhaul, modify the ground station config json file, change half of the ground stations to have a 1Gbps backhaul link and the other half to have a 2Gbps backhaul link. Then, run the main experiment.

#### Network Crash
Network crash experiment corresponds to Figure 13(b). To run the experiment, use files `experiments/network_crash.py`, `experiments/network_crash_oracle.py` and `experiments/network_crash_baseline.py`. The arguments are the same as the main experiment.

#### Distributed Ground Stations
Distributed Ground Station experiment corresponds to Table 5 and Figure 14. To run the experiment, you need to use the DGS ground station config files and bandwidth cache files. The experiment is otherwise the same as the main experiment.

The result files for the Main experiment & DGS experiment can be found [here](https://uofi.box.com/v/umbra-results). 

***The SLURM scripts:*** If you have access to SLURM and a SLURM cluster, you can run most of the experiments using the scripts in the `slurm/` folder. You will need to make sure that the path names in those scripts as well as the relay scripts (in `scripts` folder) corresponds to the intermediate files you downloaded. Even if you don't have access to SLURM, you might find some of the scripts helpful for specifying the arguments to the experiments.

### Step 4: Plotting the results
#### Main results (Table 2 & Figure 11)
- **Table 2** To generate table 2, use the code in `postprocessing/throughput.py`. You might need to change the path names in the code and run it multiple times to generate the entire table.
- **Figure 11** To generate figure 11, use the code in `postprocessing/latency.py`. You might need to change the path names in the code and run it multiple times to generate the 3 subfigures

#### Heterogeneous Backhaul (Table 4)
To generate Table 4, use `postprocessing/througput.py` to calculate the throughput for the heterogeneous backhaul experiment. 

#### Network Crash (Figure 13(b))
To generate Figure 13(b), use `postprocessing/throughput_vs_time.py` to plot the throughput vs time for the network crash experiment. You might need to change the path names in the code.

#### Distributed Ground Stations (Table 5 & Figure 14)
Generate the results of this part using the same code as the main results. You might need to change the path names in the code.

#### Microbenchmarks (Figure 12)
Figure 12 was plotted manually using the results from the main experiment. We do not have a script to generate this figure. Follow the instructions in the paper to generate the figure.
- **Figure 12(a)**: Compare the `best_flow_result.pkl` and `assignment.pkl` in each result folder. Plot the cdf of the ratios of the optimized flow rate to the link capacity for each flow. You might find `postprocessing/queue.ipynb` helpful.
- **Figure 12(b)**: Look at `maxflow_gs_queue_record.pkl` and `baseline_gs_queue_record.pkl` in each result folder of `binary_search.py`, and the `gs_queue_record.pkl` in the result folder of `smart_heuristic.py` and `basic_heuristic.py`. Plot the queue sizes of different ground stations over time.

#### Noise (Figure 13(a))
For Figure 13(a), run `experiments/binary_search.py` and specify the noise levels. Then, use `postprocessing/latency.py` to get the percentile latency for each noise level. Plot the figure manually.