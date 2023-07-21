import argparse
import datetime
import logging
import pickle

from shapely.geometry import Polygon

from simulator.types import Image

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_pickle_file', type=str, required=True)
	parser.add_argument('--output_pickle_file', default='data/image_mapping.pkl', type=str)
	parser.add_argument('--log_file', default='/dev/null', type=str)
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
	image_info = pickle.load(open(args.input_pickle_file, 'rb'))
	satellite_mapping = {}
	logger.info("Parsing image info")
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
			img = Image(300e6, image_region, image_time, id=id)
			img.score = image_time.timestamp()
			satellite_mapping[sat_id].append(img)
		except ValueError:
			logger.exception(image_info[id]['geometry']['coordinates'])
	for sat_id in satellite_mapping:
		satellite_mapping[sat_id].sort(key=lambda x: x.time)
	pickle.dump(satellite_mapping, open(args.output_pickle_file, 'wb+'))
