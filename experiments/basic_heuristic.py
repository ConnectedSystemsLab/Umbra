import argparse
import datetime
import json
import logging
import numpy as np
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from typing import List

from simulator.abstract_groundstation import AbstractGroundstation
from simulator.abstract_satellite import AbstractSatellite
from simulator.cloud import Cloud
from simulator.filters import GeographicFilter, ImageProcessingPipeline
from simulator.groundstation import BasicGroundStation
from simulator.satellite import Satellite


def simulate(
		groundstations: List[AbstractGroundstation],
		satellites: List[AbstractSatellite],
		cloud,
		bandwidth_records,
		args,
		max_step=None,
):
	"""
	Simulate the scenario with the given ground stations, satellites and clouds.
	:param max_step:
	:param bandwidth_records:
	:param groundstations:
	:param satellites:
	:param cloud:
	:return:
	"""
	simulating = True
	time_step = args.time_step
	step = 0
	current_time = args.start_time
	sat_gs_assignment_record = []
	logger = logging.getLogger("simulator")
	logger.debug("Starting simulation")
	gs_queue_length_record = defaultdict(list)
	while simulating:
		simulating = False
		groundstation_satellite_mapping = {}
		groundstation_reception = defaultdict(list)
		groundstation_indices = {}
		satellite_indices = {}
		cost_matrix = np.empty((len(groundstations) * 4, len(satellites)), dtype=float)
		bw_matrix = np.empty((len(groundstations) * 4, len(satellites)), dtype=float)
		for i, _groundstation in enumerate(groundstations):
			groundstation_indices[i] = _groundstation.get_id()
			for j, _satellite in enumerate(satellites):
				satellite_indices[j] = _satellite.id
				bandwidth = bandwidth_records[_groundstation.get_id()][_satellite.id][
					step]
				bandwidth *= args.sat_bw_multiplier
				if bandwidth > 0:
					max_size = min(bandwidth * time_step, _satellite.get_cache_size())
					if max_size == 0:
						max_size = bandwidth / 1e7  # Break even between 0 cache size and 0 bandwidth
				else:
					max_size = 0
				cost_matrix[i * 4:(i + 1) * 4, j] = -max_size
				bw_matrix[i * 4:(i + 1) * 4, j] = bandwidth
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		for i, j in zip(row_ind, col_ind):
			if bw_matrix[i][j] > 0:
				groundstation_satellite_mapping[satellite_indices[j]] = (groundstation_indices[i // 4], bw_matrix[i, j])
		logger.debug(f'gs_mapping {groundstation_satellite_mapping}')
		for i, _satellite in enumerate(satellites):
			if _satellite.id in groundstation_satellite_mapping:
				groundstation_id, bandwidth = groundstation_satellite_mapping[_satellite.id]
				next_contact = get_next_contact(_satellite.id, groundstation_id, bandwidth_records, step)
				logger.debug(f'satellite {_satellite.id} next contact {next_contact}')
				if (not next_contact) or find_gs_by_id(next_contact, groundstations).get_buffer_size() >= find_gs_by_id(
						groundstation_id, groundstations).get_buffer_size():
					_satellite.set_transmit(bandwidth)
					_satellite.update(time_step)
					transmit_queue = _satellite.get_downlink_images()
					if transmit_queue:
						for image in transmit_queue:
							image.downlink_timestamp = current_time
						logger.debug(
							f'Satellite {_satellite.id} downlink {len(transmit_queue)} images to gs {groundstation_id}')
					else:
						logger.debug(f'Satellite {_satellite.id} has no downlink images')
					groundstation_reception[groundstation_id].extend(transmit_queue)
				else:
					logger.debug(f'Satellite {_satellite.id} not downlink to gs {groundstation_id}')
					_satellite.set_transmit(0)
					_satellite.update(time_step)
			else:
				_satellite.set_transmit(0)
				_satellite.update(time_step)
			if not _satellite.is_finished():
				simulating = True
		for i, _groundstation in enumerate(groundstations):
			_groundstation.downlink_images(groundstation_reception[_groundstation.get_id()])
			_groundstation.update(time_step)
			upload_queue = _groundstation.get_upload_images()
			if upload_queue:
				logger.debug(f'GS {_groundstation.get_id()} upload {len(upload_queue)}')
				for image in upload_queue:
					image.groundstation = _groundstation.get_id()
			cloud.upload_images(deepcopy(upload_queue))
			if not _groundstation.is_finished():
				simulating = True
			gs_queue_length_record[_groundstation.get_id()].append((_groundstation.get_queue_length()))
		cloud.update(time_step)
		step += 1
		current_time += timedelta(seconds=args.time_step)
		logger.debug(f'Step {step}')
		# if not satellite_finished:
		sat_gs_assignment_record.append(groundstation_satellite_mapping)
	logger.debug(f'Simulation finished')
	return sat_gs_assignment_record, cloud, gs_queue_length_record


def get_next_contact(sat_id, gs_id, bw_record, current_step):
	while current_step < len(bw_record[gs_id][sat_id]):
		for gs in bw_record:
			if (not gs == gs_id) and bw_record[gs][sat_id][current_step] > 0:
				return str(gs)
		current_step += 1


def find_gs_by_id(id, groundstations):
	for gs in groundstations:
		if gs.get_id() == id:
			return gs
	raise KeyError("gs id not found: ", id)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_mapping_info', default='data/image_mapping.pkl', type=str)
	parser.add_argument('--gs_config', type=str, required=True)
	parser.add_argument('--start_time', type=lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'),
	                    required=True)
	parser.add_argument('--time_step', type=float, default=60)
	parser.add_argument('--cache_file', type=str, default='data/bw_cache.pkl')
	parser.add_argument('--result_dir', type=str, default='data/')
	parser.add_argument('--log_file', type=str, default='log.log')
	parser.add_argument('--sat_bw_multiplier', type=float, default=0.125)
	parser.add_argument('--iterations', type=int, default=3)
	args = parser.parse_args()
	logging.basicConfig(
		filename=args.log_file,
		level=logging.INFO,
		format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	logger = logging.getLogger('main')
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.getLogger().info(args)
	logger.info("Loading file")
	start_time = args.start_time
	gs_config = json.load(open(args.gs_config))
	groundstation_collection = []
	satellite_collection = []
	satellite_mapping = pickle.load(open(args.image_mapping_info, 'rb'))
	logger.info('Loaded satellite_mapping')
	_sat_count = 0
	for id in satellite_mapping:
		try:
			sat = Satellite(start_time, id, satellite_mapping[id])
			satellite_collection.append(sat)
			_sat_count += 1
		except Exception:
			logger.exception(f"Exception in Satellite {id}")
	for name in gs_config:
		try:
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'], name=name)
			groundstation_collection.append(gs)
		except:
			logger.exception("Exception in Ground Station")
	target_coord = []
	cloud_detector = GeographicFilter(user=None, region=Polygon(
		target_coord
	))
	cloud = Cloud(start_time, ImageProcessingPipeline([[cloud_detector]]))
	logger.info(
		f'Number of satellites: {len(satellite_collection)}, number of ground stations: {len(groundstation_collection)}')
	logger.info("Trying to get bandwidth")
	bandwidth_record = pickle.load(open(args.cache_file, 'rb'))
	logger.info("Simulating...")
	assignment, _cloud, _gs_queue_record = simulate(groundstation_collection,
	                                                satellite_collection,
	                                                cloud, bandwidth_record, args, 1440 * 5)
	throughput = sum([x[0].size for x in _cloud.image_list])
	logger.info(f"Throughput: {throughput}")
	average_delay = sum([(x[1] - x[0].time).total_seconds() for x in _cloud.image_list]) / len(_cloud.image_list)
	logger.info(f"Average delay: {average_delay}")
	Path(args.result_dir).mkdir(parents=True, exist_ok=True)
	pickle.dump(_cloud, Path(args.result_dir, 'cloud.pkl').open('wb+'))
	pickle.dump(_gs_queue_record, Path(args.result_dir, 'gs_queue_record.pkl').open('wb+'))
