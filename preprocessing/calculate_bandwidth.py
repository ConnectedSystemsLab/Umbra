import argparse
import datetime
import json
import logging
import pickle
from multiprocessing import Pool

from simulator.groundstation import BasicGroundStation
from simulator.satellite import Satellite
from simulator.satsim import get_bw_map


def get_all_satellites(filename):
	satellite_mapping = pickle.load(open(filename, 'rb'))
	satellite_collection = []
	for id in satellite_mapping:
		satellite_collection.append(Satellite(datetime.datetime.now(), id, []))
	return {sat.id: sat.read_file() for sat in satellite_collection}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--satellite_mapping_file', default='data/image_mapping.pkl', type=str)
	parser.add_argument('--gs_config', type=str, required=True)
	parser.add_argument('--start_time', type=lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'),
	                    required=True)
	parser.add_argument('--end_time', type=lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'), required=True)
	parser.add_argument('--time_step', type=float, default=60)
	parser.add_argument('--output_file', type=str, default='data/bw_cache.pkl')
	parser.add_argument('--log_file', type=str, default='/dev/null')
	parser.add_argument('--visibility_threshold', type=float, default=5)
	args = parser.parse_args()
	logging.basicConfig(
		filename=args.log_file,
		level=logging.DEBUG,
		format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
		datefmt='%H:%M:%S'
	)
	logger = logging.getLogger('main')
	logging.getLogger().addHandler(logging.StreamHandler())
	start_time = args.start_time
	end_time = args.end_time
	logger.info("Loading groundstations")
	gs_config = json.load(open(args.gs_config))
	groundstation_collection = []
	logger.info("Loading satellites")
	satellite_collection = get_all_satellites(args.satellite_mapping_file)
	for name in gs_config:
		try:
			gs = BasicGroundStation(start_time, gs_config[name]['position'], gs_config[name]['bandwidth'], name=name)
			groundstation_collection.append(gs)
		except:
			logger.exception("Exception in Ground Station")
	logger.info(
		f'Number of satellites: {len(satellite_collection)}, number of ground stations: {len(groundstation_collection)}')
	logger.info("Trying to get bandwidth")
	params = []
	for groundstation in groundstation_collection:
		for sat_id in satellite_collection:
			try:
				params.append((sat_id, satellite_collection[sat_id], groundstation.get_id(),
				               groundstation.get_position(), start_time, end_time,
				               datetime.timedelta(seconds=args.time_step), args.visibility_threshold))
				logger.info(f'Got bw for gs {groundstation.name} and sat {sat_id}')
			except:
				logger.exception(f'Error in bandwidth for gs {groundstation.name} and sat {sat_id}')
	with Pool(processes=20) as pool:
		bandwidth_record_ = dict(pool.starmap(get_bw_map, params))
	bandwidth_record = {
		gs.get_id(): {sat_id: bandwidth_record_[(gs.get_id(), sat_id,)] for sat_id in satellite_collection}
		for gs in groundstation_collection}
	pickle.dump(bandwidth_record, open(args.output_file, 'wb+'))


if __name__ == '__main__':
	main()
