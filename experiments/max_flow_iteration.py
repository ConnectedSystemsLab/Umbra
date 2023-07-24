import argparse
import datetime
import json
import logging
import math
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import List, Any

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

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
		max_flow_assignment=None,
		max_step=None,
):
	"""
	Simulate the scenario with the given ground stations, satellites and clouds.
	:param max_step:
	:param max_flow_assignment:
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
		cost_matrix = np.empty((len(groundstations), len(satellites)), dtype=float)
		bw_matrix = np.empty((len(groundstations), len(satellites)), dtype=float)
		for i, _groundstation in enumerate(groundstations):
			groundstation_indices[i] = _groundstation.get_id()
			for j, _satellite in enumerate(satellites):
				satellite_indices[j] = _satellite.id
				try:
					if not _groundstation.get_id() == max_flow_assignment[step][_satellite.id][0]:
						bandwidth = bandwidth_records[_groundstation.get_id()][_satellite.id][
							step]  # TODO interpolate this
						bandwidth *= args.sat_bw_multiplier
					else:
						bandwidth = max_flow_assignment[step][_satellite.id][1]
				except (KeyError, IndexError, TypeError):
					bandwidth = bandwidth_records[_groundstation.get_id()[:-2]][_satellite.id][
						step]  # TODO interpolate this
					bandwidth *= args.sat_bw_multiplier
				if bandwidth > 0:
					max_size = min(bandwidth * time_step, _satellite.get_cache_size())
					if max_size == 0:
						max_size = bandwidth / 1e7  # Break even between 0 cache size and 0 bandwidth
				else:
					max_size = 0
				cost_matrix[i, j] = -max_size
				bw_matrix[i, j] = bandwidth
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		for i, j in zip(row_ind, col_ind):
			if bw_matrix[i][j] > 0:
				groundstation_satellite_mapping[satellite_indices[j]] = (groundstation_indices[i], bw_matrix[i, j])
		logger.debug(f'gs_mapping {groundstation_satellite_mapping}')
		logger.debug(f'satellite_indices {satellite_indices}')
		for i, _satellite in enumerate(satellites):
			if _satellite.id in groundstation_satellite_mapping:
				groundstation_id, bandwidth = groundstation_satellite_mapping[_satellite.id]
				_satellite.set_transmit(bandwidth)
				_satellite.update(time_step)
				transmit_queue = _satellite.get_downlink_images()
				if transmit_queue:
					for image in transmit_queue:
						image.downlink_timestamp = current_time
					logger.debug(
						f'Satellite {_satellite.id} downlink {len(transmit_queue)} images to gs {groundstation_id}')
				groundstation_reception[groundstation_id].extend(transmit_queue)
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
		if max_step and step >= max_step:
			simulating = False
	logger.debug(f'Simulation finished')
	return sat_gs_assignment_record, cloud, gs_queue_length_record


def optimize(
		groundstations: List[AbstractGroundstation],
		satellites: List[AbstractSatellite],
		sat_gs_assignment_record,
		args,
) -> tuple[list[dict[Any, tuple[Any, Any]]], Any]:
	logger = logging.getLogger("optimizer")
	logger.setLevel(logging.DEBUG)
	logger.debug('Starting optimization')
	num_steps = len(sat_gs_assignment_record)
	logger.debug(f'Number of steps {num_steps}')
	flow_record = []
	sat_prev_step = {}
	gs_prev_step = {}
	graph = nx.DiGraph()
	graph.add_node('source')
	graph.add_node('cloud')
	for step in range(num_steps):
		groundstation_satellite_mapping = sat_gs_assignment_record[step]
		all_groundstations = [groundstation_satellite_mapping[sat_id][0] for sat_id in groundstation_satellite_mapping]
		for i, _groundstation in enumerate(groundstations):
			if _groundstation.get_id() in all_groundstations:
				graph.add_node(f'gs_{step}_{_groundstation.get_id()}', )
				if _groundstation.get_id() in gs_prev_step:
					graph.add_edge(
						f'gs_{gs_prev_step[_groundstation.get_id()]}_{_groundstation.get_id()}',
						f'gs_{step}_{_groundstation.get_id()}', capacity=math.inf
					)
				graph.add_edge(
					f'gs_{step}_{_groundstation.get_id()}',
					f'cloud',
					capacity=_groundstation.get_bandwidth() * args.time_step
				)
				gs_prev_step[_groundstation.get_id()] = step
			elif _groundstation.get_id() in gs_prev_step:
				graph.edges[f'gs_{gs_prev_step[_groundstation.get_id()]}_{_groundstation.get_id()}', f'cloud'][
					'capacity'] += _groundstation.get_bandwidth() * args.time_step
		for i, _satellite in enumerate(satellites):
			_satellite.update(args.time_step)
			if _satellite.id in groundstation_satellite_mapping:
				data_amout = _satellite.get_cache_size()
				_satellite.set_transmit(1e12)
				_ = _satellite.get_downlink_images()
				_groundstation, bandwidth = groundstation_satellite_mapping[_satellite.id]
				graph.add_node(f'sat_{step}_{_satellite.id}')
				if _satellite.id in sat_prev_step:
					graph.add_edge(
						f'sat_{sat_prev_step[_satellite.id]}_{_satellite.id}',
						f'sat_{step}_{_satellite.id}', capacity=math.inf)
				sat_prev_step[_satellite.id] = step
				graph.add_edge(f'source', f'sat_{step}_{_satellite.id}', capacity=data_amout)
				graph.add_edge(
					f'sat_{step}_{_satellite.id}',
					f'gs_{step}_{_groundstation}',
					capacity=bandwidth * args.time_step
				)
	logger.debug(f'Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges')
	flow_val, flow_dict = nx.maximum_flow(graph, 'source', 'cloud')
	logger.info(f'Max flow value is {flow_val}')
	for step in range(num_steps):
		groundstation_satellite_mapping = sat_gs_assignment_record[step]
		flow_assignment = {}
		for _satellite in groundstation_satellite_mapping:
			flow_assignment[_satellite] = (
				groundstation_satellite_mapping[_satellite][0],
				flow_dict[f'sat_{step}_{_satellite}'][
					f'gs_{step}_{groundstation_satellite_mapping[_satellite][0]}'] / args.time_step
			)
		flow_record.append(flow_assignment)
	logger.debug("Optimization finished")
	return flow_record, flow_val


def execute(groundstations: List[AbstractGroundstation], satellites: List[AbstractSatellite], _cloud, flow_record,
            _args,
            max_steps=None):
	execute_logger = logging.getLogger("executor")
	execute_logger.setLevel(logging.INFO)
	execute_logger.debug('Starting execution')
	num_steps = len(flow_record) if max_steps is None else max_steps
	execute_logger.debug(f'Number of steps {num_steps}')
	gs_queue_length_record = defaultdict(list)
	for _step in range(num_steps):
		if _step < len(flow_record):
			groundstation_satellite_mapping = flow_record[_step]
			groundstation_transmit_queue = defaultdict(list)
			for _satellite in satellites:
				if _satellite.id in groundstation_satellite_mapping:
					_groundstation, bandwidth = groundstation_satellite_mapping[_satellite.id]
					_satellite.set_transmit(bandwidth)
					_satellite.update(_args.time_step)
					downlink_images = _satellite.get_downlink_images()
					groundstation_transmit_queue[_groundstation].extend(downlink_images)
				else:
					_satellite.set_transmit(0)
					_satellite.update(_args.time_step)
		else:
			groundstation_transmit_queue = defaultdict(list)
			for _satellite in satellites:
				_satellite.update(_args.time_step)
		for _groundstation in groundstations:
			_groundstation.downlink_images(groundstation_transmit_queue[_groundstation.get_id()])
			_groundstation.update(_args.time_step)
			upload_queue = _groundstation.get_upload_images()
			gs_queue_length_record[_groundstation.get_id()].append((_groundstation.get_queue_length()))
			_cloud.upload_images(upload_queue)
		_cloud.update(_args.time_step)
	execute_logger.debug("Execution finished")
	return _cloud, gs_queue_length_record


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
	flow_result = None
	best_average_delay = float('inf')
	best_throughput = 0
	best_delay_cloud = None
	best_throughput_cloud = None
	baseline_cloud = None
	maxflow_cloud = None
	baseline_gs_queue_record = None
	maxflow_gs_queue_record = None
	best_throughput_gs_queue_record = None
	best_average_delay_gs_queue_record = None
	for step in range(args.iterations):
		logger.info("Simulating step {}".format(step))
		assignment, _cloud, _gs_queue_record = simulate(deepcopy(groundstation_collection),
		                                                deepcopy(satellite_collection),
		                                                deepcopy(cloud), bandwidth_record, args, flow_result, 1440 * 5)
		if baseline_cloud is None:
			baseline_cloud = _cloud
			baseline_gs_queue_record = _gs_queue_record
		simulate_throuput = sum(image[0].size for image in _cloud.image_list)
		average_delay = sum((image[1] - image[0].time).total_seconds() for image in _cloud.image_list) / len(
			_cloud.image_list)
		average_delay_on_satellite = sum(
			(image[0].downlink_timestamp - image[0].time).total_seconds() for image in _cloud.image_list) / len(
			_cloud.image_list)
		logger.info(
			f'Simulation step {step} finished, throughput: {simulate_throuput}, average delay: {average_delay}, '
			f'average delay on satellite: {average_delay_on_satellite}'
		)
		if average_delay < best_average_delay:
			best_average_delay = average_delay
			best_delay_cloud = _cloud
			best_average_delay_gs_queue_record = _gs_queue_record
			logger.info(f'New best average delay: {best_average_delay}')
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_throughput_cloud = _cloud
			best_throughput_gs_queue_record = _gs_queue_record
			logger.info(f'New best throughput: {best_throughput}')
		_flow_result, max_flow = optimize(deepcopy(groundstation_collection), deepcopy(satellite_collection),
		                                  assignment,
		                                  args)
		if flow_result is None:
			flow_result = _flow_result
		else:
			for (i, step_flow_result) in enumerate(_flow_result):
				flow_result[i].update(step_flow_result)
		logger.info(f'Optimization step {step} finished, max flow: {max_flow}')
		_cloud_new, _gs_queue_record = execute(deepcopy(groundstation_collection), deepcopy(satellite_collection),
		                                       deepcopy(cloud),
		                                       _flow_result, args)
		simulate_throuput = sum(image[0].size for image in _cloud_new.image_list)
		average_delay = sum((image[1] - image[0].time).total_seconds() for image in _cloud_new.image_list) / len(
			_cloud_new.image_list)
		logger.info(f'Execution step {step} finished, throughput: {simulate_throuput}, average delay: {average_delay}')
		if average_delay < best_average_delay:
			best_average_delay = average_delay
			best_delay_cloud = _cloud_new
			best_average_delay_gs_queue_record = _gs_queue_record
			logger.info(f'New best average delay: {best_average_delay}')
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_throughput_cloud = _cloud_new
			best_throughput_gs_queue_record = _gs_queue_record
			logger.info(f'New best throughput: {best_throughput}')
		if maxflow_cloud is None:
			maxflow_cloud = _cloud_new
			maxflow_gs_queue_record = _gs_queue_record
	Path(args.result_dir).mkdir(parents=True, exist_ok=True)
	pickle.dump(best_delay_cloud, Path(args.result_dir, 'best_delay_cloud.pkl').open('wb+'))
	pickle.dump(best_throughput_cloud, Path(args.result_dir, 'best_throughput_cloud.pkl').open('wb+'))
	pickle.dump(baseline_cloud, Path(args.result_dir, 'baseline_cloud.pkl').open('wb+'))
	pickle.dump(maxflow_cloud, Path(args.result_dir, 'maxflow_cloud.pkl').open('wb+'))
	pickle.dump(baseline_gs_queue_record, Path(args.result_dir, 'baseline_gs_queue_record.pkl').open('wb+'))
	pickle.dump(maxflow_gs_queue_record, Path(args.result_dir, 'maxflow_gs_queue_record.pkl').open('wb+'))
	pickle.dump(best_throughput_gs_queue_record,
	            Path(args.result_dir, 'best_throughput_gs_queue_record.pkl').open('wb+'))
	pickle.dump(best_average_delay_gs_queue_record,
	            Path(args.result_dir, 'best_average_delay_gs_queue_record.pkl').open('wb+'))
# logger.info(f'Evaluation result: {cloud.evaluate()}')
# pickle.dump(cloud, open(args.result_file, 'wb+'))
