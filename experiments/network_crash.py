import argparse
import datetime
import json
import logging
import math
import networkx as nx
import numpy as np
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from typing import List, Any

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
		bw_offset=0,
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
	snapshot = None
	while simulating:
		simulating = False
		if step == crash_step:
			snapshot = deepcopy([groundstations, satellites, cloud])
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
				try:
					if not _groundstation.get_id() == max_flow_assignment[step][_satellite.id][0]:
						bandwidth = bandwidth_records[_groundstation.get_id()[:-2]][_satellite.id][
							step + bw_offset]
						bandwidth *= args.sat_bw_multiplier
					else:
						bandwidth = max_flow_assignment[step][_satellite.id][1]
				except (KeyError, IndexError, TypeError):
					bandwidth = bandwidth_records[_groundstation.get_id()][_satellite.id][
						step + bw_offset]
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
			if step >= crash_step and _groundstation.get_id() == crash_gs:
				continue
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
	return sat_gs_assignment_record, cloud, snapshot


def optimize(
		groundstations: List[AbstractGroundstation],
		satellites: List[AbstractSatellite],
		sat_gs_assignment_record,
		args,
		initial_condition=None
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
	if initial_condition is not None:
		sat_init, gs_init = initial_condition
		for sat_id in sat_init:
			graph.add_node(f'sat_0_{sat_id}')
			graph.add_edge('source', f'sat_0_{sat_id}', capacity=sat_init[sat_id])
			sat_prev_step[sat_id] = 0
		for gs_id in gs_init:
			graph.add_node(f'gs_0_{gs_id}')
			graph.add_edge(f'gs_0_{gs_id}', 'cloud', capacity=gs_init[gs_id])
			gs_prev_step[gs_id] = 0
	for step in range(num_steps):
		if step >= crash_step:
			for _groundstation in groundstations:
				if _groundstation.get_id() in crash_gs:
					groundstations.remove(_groundstation)
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
            max_steps=None, ):
	snapshot = []
	execute_logger = logging.getLogger("executor")
	execute_logger.setLevel(logging.INFO)
	execute_logger.debug('Starting execution')
	num_steps = len(flow_record) if max_steps is None else max_steps
	execute_logger.debug(f'Number of steps {num_steps}')
	gs_queue_length_record = defaultdict(list)
	for _step in range(num_steps):
		if _step == crash_step:
			snapshot = [deepcopy(groundstations), deepcopy(satellites), deepcopy(_cloud)]
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
	return _cloud, snapshot


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_mapping_info', default='data/image_mapping.pkl', type=str)
	parser.add_argument('--gs_config', type=str, default='data/gs_config_1G.json')
	parser.add_argument('--start_time', type=lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'),
	                    required=True)
	parser.add_argument('--time_step', type=float, default=60)
	parser.add_argument('--cache_file', type=str, default='data/bw_cache.pkl')
	parser.add_argument('--result_dir', type=str, default='data/network_crash')
	parser.add_argument('--log_file', type=str, default='log/network_crash.log')
	parser.add_argument('--sat_bw_multiplier', type=float, default=0.125)
	args = parser.parse_args()
	logging.basicConfig(
		filename=args.log_file,
		level=logging.INFO,
		format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	crash_step = 3600
	crash_gs = ['1', '3', '7', '10']
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
	flow_result = None
	best_throughput = 0
	best_cloud = None
	baseline_cloud = None
	best_status = None
	baseline_gs_queue_record = None
	best_gs_queue_record = None
	best_snapshot = []
	logger.info("Simulating Phase 1")
	for step in range(2):
		logger.info("Simulating step {}".format(step))
		assignment, _cloud, snapshot = simulate(
			deepcopy(groundstation_collection),
			deepcopy(satellite_collection),
			deepcopy(cloud), bandwidth_record,
			args, flow_result,
			7200
		)
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
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_snapsot = snapshot
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
		_cloud_new, snapshot = execute(deepcopy(groundstation_collection),
		                               deepcopy(satellite_collection),
		                               deepcopy(cloud),
		                               _flow_result, args, max_steps=7200)
		simulate_throuput = sum(image[0].size for image in _cloud_new.image_list)
		average_delay = sum((image[1] - image[0].time).total_seconds() for image in _cloud_new.image_list) / len(
			_cloud_new.image_list)
		logger.info(f'Execution step {step} finished, throughput: {simulate_throuput}, average delay: {average_delay}')
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_cloud = _cloud_new
			best_snapsot = snapshot
			logger.info(f'New best throughput: {best_throughput}')
	initial_condition = [
		{sat.id: sat.get_cache_size() for sat in best_snapsot[1]},
		{gs.get_id(): gs.get_buffer_size() * 300e6 for gs in best_snapsot[0] if gs.get_id() not in crash_gs}
	]
	logger.info('Simulating Phase 2')
	best_throughput = 0
	flow_result = None
	intermediate_gs, intermediate_sat, intermediate_cloud = best_snapsot
	intermediate_gs = [gs for gs in intermediate_gs if gs.get_id() not in crash_gs]
	for step in range(2):
		logger.info("Simulating step {}".format(step))
		assignment, _cloud, _ = simulate(
			deepcopy(intermediate_gs),
			deepcopy(intermediate_sat),
			deepcopy(intermediate_cloud),
			bandwidth_record,
			args, flow_result,
			crash_step, bw_offset=crash_step
		)
		simulate_throuput = sum(image[0].size for image in _cloud.image_list)
		average_delay = sum((image[1] - image[0].time).total_seconds() for image in _cloud.image_list) / len(
			_cloud.image_list)
		logger.info(
			f'Simulation step {step} finished, throughput: {simulate_throuput}, average delay: {average_delay}'
		)
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_cloud = _cloud
			logger.info(f'New best throughput: {best_throughput}')
		_flow_result, max_flow = optimize(
			deepcopy(intermediate_gs),
			deepcopy(intermediate_sat),
			assignment,
			args,
			initial_condition
		)
		if flow_result is None:
			flow_result = _flow_result
		else:
			for (i, step_flow_result) in enumerate(_flow_result):
				flow_result[i].update(step_flow_result)
		logger.info(f'Optimization step {step} finished, max flow: {max_flow}')
		_cloud_new, _ = execute(
			deepcopy(intermediate_gs),
			deepcopy(intermediate_sat),
			deepcopy(intermediate_cloud),
			_flow_result,
			args,
			max_steps=crash_step
		)
		simulate_throuput = sum(image[0].size for image in _cloud_new.image_list)
		average_delay = sum((image[1] - image[0].time).total_seconds() for image in _cloud_new.image_list) / len(
			_cloud_new.image_list)
		logger.info(f'Execution step {step} finished, throughput: {simulate_throuput}, average delay: {average_delay}')
		if simulate_throuput > best_throughput:
			best_throughput = simulate_throuput
			best_cloud = _cloud_new
			logger.info(f'New best throughput: {best_throughput}')

	Path(args.result_dir).mkdir(parents=True, exist_ok=True)
	pickle.dump(best_cloud, Path(args.result_dir, 'best_throughput_cloud.pkl').open('wb+'))
