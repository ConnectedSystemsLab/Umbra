import argparse
import datetime
import json
import logging
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from simulator.cloud import Cloud
from simulator.filters import GeographicFilter, ImageProcessingPipeline
from simulator.groundstation import BasicGroundStation
from simulator.satellite import Satellite
from simulator.satsim import get_bw_map
from simulator.types import Image

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_info', type=str, required=True)
	parser.add_argument('--image_mapping_info', default='data/image_mapping.pkl', type=str)
	parser.add_argument('--gs_config', type=str, required=True)
	parser.add_argument('--start_time', type=float, required=True)
	parser.add_argument('--time_step', type=float, default=.2)
	parser.add_argument('--cache_file', type=str, default='data/bw_cache.pkl')
	parser.add_argument('--result_file', type=str, default='data/cloud.pkl')
	parser.add_argument('--log_file', type=str, default='log.log')
	parser.add_argument('--sat_bw_multiplier', type=float, default=0.125)
	parser.add_argument('--load_balancing', action='store_true')
	parser.add_argument('--alpha', type=float, default=2.0)
	parser.set_defaults(load_balancing=False)
	args = parser.parse_args()
	logging.basicConfig(
		filename=args.log_file,
		level=logging.DEBUG,
		format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	logger = logging.getLogger('main')
	logging.getLogger().addHandler(logging.StreamHandler())
	logger.info("Loading file")
	start_time = datetime.datetime.fromtimestamp(args.start_time)
	image_info = pickle.load(open(args.image_info, 'rb'))
	gs_config = json.load(open(args.gs_config))
	groundstation_collection = []
	satellite_collection = []
	satellite_mapping = {}
	logger.info("Parsing image info")
	try:
		satellite_mapping = pickle.load(open(args.image_mapping_info, 'rb'))
		logger.info('Loaded satellite_mapping`')
	except FileNotFoundError:
		for id in image_info:
			sat_id = image_info[id]['properties']['satellite_id']
			image_time = image_info[id]['properties']['acquired']
			try:
				image_time = datetime.datetime.strptime(image_time, '%Y-%m-%dT%H:%M:%S.%fZ')
			except ValueError:
				image_time = datetime.datetime.strptime(image_time, '%Y-%m-%dT%H:%M:%SZ')
			coordinates = image_info[id]['geometry']['coordinates'][0]
			if len(coordinates) == 1:
				coordinates = coordinates[0]
			try:
				image_region = Polygon(coordinates)
				if sat_id not in satellite_mapping:
					satellite_mapping[sat_id] = []
				img = Image(335e6, image_region, image_time)
				satellite_mapping[sat_id].append(img)
			except ValueError:
				logger.exception(image_info[id]['geometry']['coordinates'])
		pickle.dump(satellite_mapping, open(args.image_mapping_info, 'wb+'))
	_sat_count = 0
	for id in satellite_mapping:
		try:
			sat = Satellite(start_time, id, satellite_mapping[id])
			satellite_collection.append(sat)
			_sat_count += 1
		# if _sat_count > 10:
		# 	break
		except Exception:
			logger.exception(f"Exception in Satellite {id}")
	for name in gs_config:
		try:
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'] / 4,
			                        name=name + '_1')
			groundstation_collection.append(gs)
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'] / 4,
			                        name=name + '_2')
			groundstation_collection.append(gs)
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'] / 4,
			                        name=name + '_3')
			groundstation_collection.append(gs)
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'] / 4,
			                        name=name + '_4')
			groundstation_collection.append(gs)
		except:
			logger.exception("Exception in Ground Station")
	target_coord = [
		[
			-123.50555419921875,
			41.20345619205131
		],
		[
			-123.51654052734374,
			41.135227480564936
		],
		[
			-123.53302001953124,
			41.03585891144301
		],
		[
			-123.48632812499999,
			40.98197154086656
		],
		[
			-123.42041015624999,
			40.977824533189505
		],
		[
			-123.45336914062499,
			40.93426521177941
		],
		[
			-123.49456787109375,
			40.86991083161536
		],
		[
			-123.52752685546875,
			40.81380923056958
		],
		[
			-123.4918212890625,
			40.749337730454826
		],
		[
			-123.53027343749999,
			40.701463603604594
		],
		[
			-123.58245849609375,
			40.65563874006118
		],
		[
			-123.39569091796875,
			40.37584377696013
		],
		[
			-123.26385498046874,
			40.283716270542584
		],
		[
			-123.13751220703125,
			40.18726672309203
		],
		[
			-123.14300537109374,
			40.09908414736847
		],
		[
			-123.21716308593749,
			40.107487419012415
		],
		[
			-123.2281494140625,
			40.0759697987031
		],
		[
			-123.12927246093751,
			39.95606977009003
		],
		[
			-123.06335449218749,
			39.87391156801293
		],
		[
			-123.01666259765624,
			39.918162846609455
		],
		[
			-122.92877197265625,
			39.9602803542957
		],
		[
			-122.87109375,
			40.03812939078128
		],
		[
			-122.75024414062499,
			40.113789191575236
		],
		[
			-122.772216796875,
			40.147388783540485
		],
		[
			-122.838134765625,
			40.20614809577503
		],
		[
			-122.87933349609376,
			40.300476079749494
		],
		[
			-122.92053222656249,
			40.33188951824973
		],
		[
			-122.93701171874999,
			40.35700974577561
		],
		[
			-122.904052734375,
			40.44067626268237
		],
		[
			-122.88482666015625,
			40.46784549077255
		],
		[
			-122.92327880859374,
			40.51171103483292
		],
		[
			-122.93975830078124,
			40.56389453066509
		],
		[
			-122.991943359375,
			40.5930995321649
		],
		[
			-123.05511474609375,
			40.59935608796518
		],
		[
			-123.03314208984374,
			40.64938745451835
		],
		[
			-123.02215576171875,
			40.71603763556807
		],
		[
			-122.96722412109374,
			40.8034148344062
		],
		[
			-122.89031982421874,
			40.84706035607122
		],
		[
			-122.82440185546875,
			40.92388970852945
		],
		[
			-122.75848388671875,
			41.03585891144301
		],
		[
			-122.72552490234375,
			41.11039942586733
		],
		[
			-122.69805908203125,
			41.18072118284585
		],
		[
			-122.67333984374999,
			41.253032440653186
		],
		[
			-122.772216796875,
			41.288126204331704
		],
		[
			-122.8656005859375,
			41.236511201246216
		],
		[
			-122.98095703125,
			41.22205169039092
		],
		[
			-123.00292968749999,
			41.176586696571015
		],
		[
			-123.0908203125,
			41.1290213474951
		],
		[
			-123.07708740234374,
			41.091772220976644
		],
		[
			-123.14849853515625,
			41.10005163093046
		],
		[
			-123.24188232421875,
			41.147637985391874
		],
		[
			-123.32427978515625,
			41.168316941075766
		],
		[
			-123.3819580078125,
			41.19725651800892
		],
		[
			-123.46710205078124,
			41.21172151054787
		],
		[
			-123.50555419921875,
			41.20345619205131
		],
	]
	cloud_detector = GeographicFilter(user=None, region=Polygon(
		target_coord
	))
	cloud = Cloud(start_time, ImageProcessingPipeline([[cloud_detector]]))
	logger.info(
		f'Number of satellites: {len(satellite_collection)}, number of ground stations: {len(groundstation_collection)}')
	logger.info("Trying to get bandwidth")
	if Path(args.cache_file).exists():
		bandwidth_record = pickle.load(open(args.cache_file, 'rb'))
	else:
		bandwidth_record = defaultdict(dict)
		for groundstation in groundstation_collection:
			for satellite in satellite_collection:
				try:
					bandwidth_record[groundstation.get_id()][satellite.id] = get_bw_map(
						satellite, groundstation, start_time, start_time + datetime.timedelta(days=40),
						delta=datetime.timedelta(minutes=1)
					)
					logger.info(f'Got bw for gs {groundstation.name} and sat {satellite.id}')
				except:
					logger.exception(f'Error in bandwidth for gs {groundstation.name} and sat {satellite.id}')
		pickle.dump(bandwidth_record, open(args.cache_file, 'wb+'))
	if args.load_balancing:
		logger.info('Computing mean contact distance for load balancing')
		for satellite in satellite_collection:
			bw_trace = None
			for groundstation in groundstation_collection:
				if bw_trace is None:
					bw_trace = deepcopy(bandwidth_record[groundstation.get_id()][satellite.id])
				else:
					bw_trace = np.maximum(bw_trace, bandwidth_record[groundstation.get_id()][satellite.id])
			distances = []
			counter = 0
			for i, bw in enumerate(bw_trace):
				if bw == 0:
					counter += 1
				elif not counter == 0:
					distances.append(counter)
					counter = 0
			satellite.mean_contact_distance = np.mean(distances)

	logger.info("Simulating...")
	simulating = True
	time_step = args.time_step
	step = 0
	current_time = start_time
	while simulating:
		simulating = False
		groundstation_satellite_mapping = {}
		groundstation_reception = defaultdict(list)
		groundstation_indices = {}
		satellite_indices = {}
		cost_matrix = np.empty((len(groundstation_collection), len(satellite_collection)), dtype=float)
		bw_matrix = np.empty((len(groundstation_collection), len(satellite_collection)), dtype=float)
		for i, groundstation in enumerate(groundstation_collection):
			groundstation_indices[i] = groundstation.get_id()
			for j, satellite in enumerate(satellite_collection):
				satellite_indices[satellite.id] = j
				bandwidth = bandwidth_record[groundstation.get_id()[:-2]][satellite.id][step]  # TODO interpolate this
				bandwidth *= args.sat_bw_multiplier
				max_size = min(bandwidth * time_step, satellite.get_cache_size())
				cost_matrix[i, j] = -max_size
				bw_matrix[i, j] = bandwidth
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		for i, j in zip(row_ind, col_ind):
			groundstation_satellite_mapping[j] = (i, bw_matrix[i, j])
		logger.debug(f'gs_mapping {groundstation_satellite_mapping}')
		logger.debug(f'satellite_indices {satellite_indices}')
		for i, satellite in enumerate(satellite_collection):
			if satellite_indices[satellite.id] in groundstation_satellite_mapping:
				groundstation_id, bandwidth = groundstation_satellite_mapping[satellite_indices[satellite.id]]
				if args.load_balancing:
					current_groundstation = None
					for groundstation in groundstation_collection:
						if groundstation.get_id() == groundstation_id:
							current_groundstation = groundstation
					expected_delay_time = current_groundstation.get_buffer_size() / current_groundstation.get_bandwidth()
					if expected_delay_time > satellite.mean_contact_distance * args.alpha:
						logger.info(
							f'Satellite {satellite.id} chose not to downlink to gs {current_groundstation.get_id()}')
						continue
				groundstation_id = groundstation_indices[groundstation_id]
				satellite.set_transmit(bandwidth)
				satellite.update(time_step)
				transmit_queue = satellite.get_downlink_images()
				if transmit_queue:
					for image in transmit_queue:
						image.downlink_timestamp = current_time
					logger.info(
						f'Satellite {satellite.id} downlink {len(transmit_queue)} images to gs {groundstation_id}')
				groundstation_reception[groundstation_id].extend(deepcopy(transmit_queue))
			else:
				satellite.set_transmit(0)
				satellite.update(time_step)
			if not satellite.is_finished():
				simulating = True
		for i, groundstation in enumerate(groundstation_collection):
			groundstation.downlink_images(groundstation_reception[groundstation.get_id()])
			groundstation.update(time_step)
			upload_queue = groundstation.get_upload_images()
			if upload_queue:
				logger.info(f'GS {groundstation.get_id()} upload {len(upload_queue)}')
				for image in upload_queue:
					image.groundstation = groundstation.get_id()
			cloud.upload_images(deepcopy(upload_queue))
			if not groundstation.is_finished():
				simulating = True
		cloud.update(time_step)
		step += 1
		current_time += timedelta(seconds=args.time_step)
		logger.info(f'Step {step}')
	logger.info(f'Evaluation result: {cloud.evaluate()}')
	pickle.dump(cloud, open(args.result_file, 'wb+'))
