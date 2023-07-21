from datetime import datetime, timedelta
from typing import List

import itur
import numpy as np

from . import satnog_utils as utils
from .weatherservice import WeatherService


def get_snr(dist, precip, freq, el, lat, long):
	fsl = 20 * np.log10(dist) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / 3e8)
	Ag, Ac, Ar, As, A = itur.atmospheric_attenuation_slant_path(lat=lat, lon=long, f=freq / 1e9, el=el, p=0.01, D=1,
	                                                            R001=precip, return_contributions=True)
	const = 8.2 + 29 + 228.6 - 78.853
	return const - (fsl + A.value)


def snr_to_datarate(snr):
	rate = 0
	rate_vals = [2 * 1 / 4, 2 * 1 / 3, 2 * 2 / 5, 2 * 1 / 2, 2 * 3 / 5, 2 * 2 / 3, 2 * 3 / 4, 2 * 4 / 5, 2 * 5 / 6,
	             2 * 8 / 9, 2 * 9 / 10, 3 * 3 / 5, 3 * 2 / 3, 3 * 3 / 4, 3 * 5 / 6, 3 * 8 / 9, 3 * 9 / 10, 4 * 2 / 3,
	             4 * 3 / 4, 4 * 4 / 5, 4 * 5 / 6, 4 * 8 / 9, 4 * 9 / 10, 5 * 3 / 4, 5 * 4 / 5, 5 * 5 / 6, 5 * 8 / 9,
	             5 * 9 / 10]
	snr_vals = [-2.35, -1.24, -0.30, 1.00, 2.23, 3.10, 4.03, 4.68, 5.18, 6.20, 6.42, 5.50, 6.62, 7.91, 9.35, 10.69,
	            10.98, 8.97, 10.21, 11.03, 11.61, 12.89, 13.13, 12, 73, 13.64, 14.28, 15.69, 16.05]
	for idx in range(len(rate_vals)):
		if snr > snr_vals[idx]:
			cur_rate = rate_vals[idx]
			if cur_rate > rate:
				rate = cur_rate
	return 76.8 * 1e6 * rate * 6


def get_weather_from_hourly(timestamps, precips, ts):
	if len(timestamps) == 0:
		return 0
	idx_set = 0
	for idx in range(len(timestamps) - 1):
		if timestamps[idx] <= ts < timestamps[idx + 1]:
			cur_idx = idx
			idx_set = 1
	if idx_set == 0:
		if ts < timestamps[0]:
			return precips[0]
		if ts > timestamps[-1]:
			return precips[-1]

	w = (ts - timestamps[cur_idx]) / (timestamps[cur_idx + 1] - timestamps[cur_idx])
	p = (1 - w) * precips[cur_idx] + w * precips[cur_idx + 1]
	return p


def __get_bw_map__(tle: List[str], ground_pos: List[float], start_time: datetime, end_time: datetime, delta: timedelta,
                   visibility_thresh=15):
	ws = WeatherService('data/cache/weather')
	cur_gs = {
		'lat': ground_pos[0],
		'lng': ground_pos[1],
		'altitude': ground_pos[2]
	}
	num_vals = int(np.ceil((end_time - start_time) / delta))
	print(start_time, end_time, end_time - start_time, num_vals)
	az, el, dist, availability = utils.get_availability_list(tle, cur_gs, start_time, end_time, delta)
	# Some weather shenanigans to get fine-grained weather data Basically, we are making one weather call per day,
	# which returns hourly weather for that day But we interpolate to get fine-grained precipitation data Note this
	# assumes, we are interested in at most a day. If you are interested in multiple days of data, then you want to
	# loop over this by changing the start_time variable in increments of a day.
	timestamps, precips = [], []
	weather_start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
	weather_end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
	cur_time = weather_start_time
	while cur_time <= weather_end_time:
		try:
			cur_weather = \
				ws.get_weather_info(cur_gs['lat'], cur_gs['lng'], cur_time.isoformat().replace('+00:00', 'Z'))['hourly'][
					'data']
		except:
			cur_weather = []
		_timestamps = [cw['time'] for cw in cur_weather]
		_precips = [(cw['precipIntensity'] if 'percipIntensity' in cw else 0) for cw in cur_weather]
		timestamps.extend(_timestamps)
		precips.extend(_precips)
		cur_time += timedelta(days=1)

	# timestamps = [cw['time'] for cw in cur_weather]
	# precips = [cw['precipIntensity'] for cw in cur_weather]
	# timestamps_next = [nw['time'] for nw in next_weather]
	# precips_next = [nw['precipIntensity'] for nw in next_weather]
	# timestamps = timestamps + timestamps_next
	# precips = precips + precips_next

	# Compute fine-grained precipitation
	lats = np.zeros(dist.shape) + cur_gs['lat']
	lngs = np.zeros(dist.shape) + cur_gs['lng']

	r001 = np.zeros(dist.shape)
	for idx in range(r001.shape[0]):
		r001[idx] = get_weather_from_hourly(timestamps, precips, (start_time + idx * delta).timestamp()) + 1e-6
	print('Got hourly weather')
	# Get SNR and datarate from the old model for elevation>15 degrees

	correct_idx = el > visibility_thresh
	snrs = get_snr(dist[correct_idx] * 1000, r001[correct_idx], 8e9, el[correct_idx], lats[correct_idx],
	               lngs[correct_idx])
	datarates = np.array([snr_to_datarate(snr) for snr in snrs])
	snr_map = np.zeros(dist.shape)
	datarate_map = np.zeros(dist.shape)
	snr_map[correct_idx] = snrs
	datarate_map[correct_idx] = datarates

	# commit weather service
	ws.commit()
	return datarate_map.tolist()


def get_bw_map(satellite_id: str, satellite_tle: List[str], groundstation_id: str, groundstation_position: List[float],
               start_time: datetime, end_time: datetime, delta: timedelta, visibility_thresh=15):
	result = __get_bw_map__(
		satellite_tle,
		groundstation_position,
		start_time, end_time, delta, visibility_thresh=visibility_thresh
	)
	print(f'Got BW map for {satellite_id} at {groundstation_id}')
	return (groundstation_id, satellite_id,), result
