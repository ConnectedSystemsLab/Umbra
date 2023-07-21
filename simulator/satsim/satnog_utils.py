import os

import cv2
import numpy as np
from astropy import time
from astropy.coordinates import EarthLocation
from pycraf import satellite


def image_from_rect(image, rect):
	return image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def get_in_image_frame(idx, ref_rect):
	# Convert coordinates from a rectangle frame to an image frame
	idx_out = np.zeros(idx.shape)
	idx_out[:, 0] = idx[:, 0] + ref_rect[0]
	idx_out[:, 1] = idx[:, 1] + ref_rect[1]
	return idx_out


def get_in_rect_frame(idx, ref_rect):
	idx_out = np.zeros(idx.shape)
	idx_out[:, 0] = idx[:, 0] - ref_rect[0]
	idx_out[:, 1] = idx[:, 1] - ref_rect[1]
	return idx_out


def resize_rect_for_digit(digit_rect, image):
	cur_rect_pad = [digit_rect[0] - 4, digit_rect[1] - 2, digit_rect[2] + 8, digit_rect[3] + 4]
	digit_img = np.zeros([digit_rect[3] + 4, digit_rect[2] + 8], dtype=np.uint8)
	digit_img[2:2 + digit_rect[3], 4:4 + digit_rect[2]] = image[digit_rect[1]:digit_rect[1] + digit_rect[3],
														  digit_rect[0]:digit_rect[0] + digit_rect[2]]
	roi = cv2.resize(digit_img, (28, 28))
	return roi


def number_from_digit_list(digits, rect_list):
	num = 0
	is_suspect = False
	is_negative = False
	centers = [rect[1] + rect[3] // 2 for rect in rect_list]
	if ((len(rect_list) > 4) or (np.max(centers) - np.min(centers) > 2)):
		is_suspect = True
	else:
		rect_hz = np.array([rect[0] for rect in rect_list])
		sort_idx = np.argsort(rect_hz)
		digits = digits[sort_idx]
		for i in range(len(digits)):
			if (rect_list[i][2] < 4 and rect_list[i][3] < 4):
				is_suspect = True
			if (i == 0 and digits[i] == -1):
				is_negative = True
			elif (digits[i] < 0):
				is_suspect = True
			else:
				num = num + digits[i] * (10 ** (len(digits) - i - 1))
	if is_negative:
		num = num * -1
	return (num, is_suspect)


def filter_obsrvs(obsrv_list, filter_str, filter_val):
	obsrv_list_out = []
	if (obsrv_list):
		assert type(obsrv_list[0][filter_str]) == int
		vals = np.array([obsrv[filter_str] for obsrv in obsrv_list])
		idx = np.where(vals == filter_val)[0]
		obsrv_list_out = [obsrv_list[cur_idx] for cur_idx in idx]
	return obsrv_list_out


def get_file(user, ip, fn):
	if (not os.path.exists(fn)):
		cmd = 'scp ' + str(user) + '@' + str(ip) + ':' + str(fn) + ' ' + str(fn)
		print(cmd)
		os.system(cmd)


def fractional_day(ts):
	return ts.timetuple().tm_yday + ts.timetuple().tm_hour / 24 + ts.timetuple().tm_min / 24 / 60


def get_availability_list(tle_lines, gs, start_time, end_time, delta):
	# Get satellite-ground station viability for a satellite-ground station pair
	# tle_lines present tle vals read from a tle file
	# gs: ground station observation from the database
	# start_time, end_time, delta: time values to get the availability for
	num_vals = int(np.ceil((end_time - start_time).total_seconds() / delta.total_seconds()))
	availability = np.zeros((num_vals))
	az = np.zeros((num_vals))
	el = np.zeros((num_vals))
	dist = np.zeros((num_vals))
	days = np.array([np.float_(line[20:32]) for line in tle_lines[0::2]])
	years = np.array([np.int_(line[18:20])+2000 for line in tle_lines[0::2]])
	ts = np.array([start_time + i * delta for i in range(num_vals)])
	#tle_idx = [np.where(days < fractional_day(cur_time))[0][-1] for cur_time in ts]
	tle_idx = [np.where(np.logical_and(days < fractional_day(cur_time), years==cur_time.timetuple().tm_year))[0][-1] for cur_time in ts]
	obstime = time.Time(ts)
	gs_loc = [gs['lng'], gs['lat'], gs['altitude']]
	location = EarthLocation.from_geodetic(gs_loc[0], gs_loc[1], gs_loc[2])
	sat_obs = satellite.SatelliteObserver(location)
	v = np.unique(tle_idx)
	for idx in range(len(v)):
		cur_idx = np.where(tle_idx == v[idx])[0]
		tle1 = tle_lines[v[idx] * 2]
		tle2 = tle_lines[v[idx] * 2 + 1]
		az[cur_idx], el[cur_idx], dist[cur_idx] = sat_obs.azel_from_sat("\n" + tle1 + tle2, obstime[cur_idx])
	availability = el > 0

	return az, el, dist, availability


def get_availability_satnogs(obsrv_start_times, obsrv_end_times, start_time, end_time, delta):
	num_vals = int(np.ceil((end_time - start_time).total_seconds() / delta.total_seconds()))
	ts = np.array([start_time + i * delta for i in range(num_vals)])
	availability_dset = [
		np.sum(np.logical_and(cur_time.timestamp() < obsrv_end_times, cur_time.timestamp() > obsrv_start_times)) for
		cur_time in ts]
	return np.array(availability_dset)
