import decimal
import heapq
import os
import sys
from datetime import datetime
from datetime import timedelta
from heapq import heappush
from typing import List

import numpy as np
import requests
from pyorbital.orbital import Orbital

from .abstract_satellite import AbstractSatellite
from .satsim.satnog_utils import fractional_day

sys.path.insert(0, os.getcwd() + "/../")
from simulator.types import Image


class Satellite(AbstractSatellite):
	def __init__(self, start_time: datetime, id: str, images: List[Image]):
		super().__init__(start_time, id, images)
		self.date = start_time  # Current time on satellite
		# self.sat_name = self.get_name()  # Name of satellite
		self.norad = self.get_norad()  # NORAD ID of satellite
		self.latest_TLE = self.get_latest_TLE()  # Lastest TLE of Satellite at Real Time
		self.TLE_file = self.read_file()  # Historical TLEs of the satellite
		self.bandwidth = 0  # Bandwidth to transmit data
		self.contact_time = 10  # Time Satellite has to transmit to groundstation
		self.cache = []  # List of Images available to transmit respecting the current time at the satellite
		self.cache_size = 0  # Size of the cache
		self.populate_cache()

	def get_norad(self):
		self.id = self.id.lower()

		filename = "data/names.txt"
		translations = {}
		with open(filename) as f:
			lines = f.readlines()
		for each in lines:
			line = each.split()
			if line[0] == "0505" or line[0] == "0711":
				translations[line[0]] = line[6]
			elif line[0] != '#' and "SKYSAT" not in line and "OBJECT" not in line:
				translations[line[0]] = line[7]
		norad = translations[self.id]
		return norad

	# Returns the latest TLE of the satellite from Celestrak
	def get_latest_TLE(self):
		request_CT = "https://celestrak.com/NORAD/elements/gp.php?CATNR=" + self.norad + "&FORMAT=TLE"
		session = requests.Session()
		TLE = session.get(request_CT).text.splitlines()[1:]
		return TLE

	# Reads the historical TLE files
	def read_file(self):
		f = open("data/HistoricalTLEs/sat" + self.norad + ".txt", "r")
		return [line for line in f.readlines() if line.strip()]

	# Populates the cache respecting the current time
	def populate_cache(self):
		while len(self.images) > 0 and self.images[0].time <= self.date:
			img = self.images.pop(0)
			heappush(self.cache, (img.score, img))

	# Returns True if images are left to be transmitted at a given point in time
	def has_image(self):
		if self.cache is None:
			return False
		elif len(self.cache) > 0:
			return True
		else:
			return False

	# Updates the time of the satellite by adding seconds to the existing time
	def update(self, time):
		# Time in seconds to update the current time at the satellite
		self.date = self.date + timedelta(seconds=time)
		while len(self.images) > 0 and self.images[0].time <= self.date:
			img = self.images.pop(0)
			heappush(self.cache, (img.score, img))
			self.cache_size += img.size
		self.contact_time = time

	# Returns the minimum of the data available on the satellite to transmit and bandwidth * contact time
	def get_cache_size(self):
		return self.cache_size

	# Sets the bandwidth to send data on the satellite
	def set_transmit(self, bandwidth):
		# Bandwith to update the bandwidth at which the satellite transmits data
		self.bandwidth = bandwidth

	# Used for comparision of the day in get_position and get_tle
	def fractional_day(self, ts):
		return ts.timetuple().tm_yday + ts.timetuple().tm_hour / 24 + ts.timetuple().tm_min / 24 / 60

	# Returns the position of the satellite at its current time
	def get_position(self):
		mydate = self.date
		TLE_file = self.TLE_file
		norad = self.norad
		days = np.array([np.float_(line[16:32]) for line in TLE_file[0:-1:2]])

		f_date = str(fractional_day(mydate))
		whole = f_date.split('.')[0]
		dec = f_date.split('.')[1]
		if len(whole) == 1:
			whole = '00' + whole
		elif len(whole) == 2:
			whole = '0' + whole
		elif len(whole) == 3:
			whole = whole
		str_f = str(mydate.year)[-2:] + whole + '.' + dec
		f_date = decimal.Decimal(str_f)
		tle_idx = [np.where(days < f_date)[0][-1]]
		v = np.unique(tle_idx)
		tles = []
		for idx in range(len(v)):
			tle1 = TLE_file[v[idx] * 2]
			tle2 = TLE_file[v[idx] * 2 + 1]
			tles.append(tle1)
			tles.append(tle2)
		sat = Orbital(norad, line1=tles[0], line2=tles[1])
		lon, lat, alt = sat.get_lonlatalt(mydate)
		return [lat, lon, alt * 1000]

	# Downlinks images to groundstation
	def get_downlink_images(self):
		total = 0
		if self.cache != None:
			for each in self.cache:
				total += each[1].size
		to_send = []
		if total > (self.bandwidth * self.contact_time):
			temp = 0
			while self.cache:
				each = heapq.heappop(self.cache)
				score = each[0]
				each = each[1]
				if (temp + each.size) <= (self.bandwidth * self.contact_time):
					temp += each.size
					to_send.append(each)
					self.cache_size -= each.size
				else:
					heapq.heappush(self.cache, (score, each))
					break
		else:
			while self.cache:
				to_send.append(heapq.heappop(self.cache)[1])
			self.cache = []
			self.cache_size = 0
		self.bandwidth = 0
		self.contact_time = 0
		return to_send

	# Returns True if no more images are left to transmit till the end of time
	def is_finished(self):
		# code.InteractiveConsole()
		if len(self.cache) > 0:
			return False
		else:
			if self.images == None:
				return True
			elif len(self.images) > 0:
				return False
			else:
				return True

	# Returns the TLE of the satellite at its current time
	def get_tle(self):
		days = np.array([np.float_(line[20:32]) for line in self.TLE_file[0:-1:2]])
		tle_idx = [np.where(days < self.fractional_day(self.date))[0][-1]]
		v = np.unique(tle_idx)
		tles = []
		for idx in range(len(v)):
			tle1 = self.TLE_file[v[idx] * 2]
			tle2 = self.TLE_file[v[idx] * 2 + 1]
			tles.append(tle1)
			tles.append(tle2)
		if len(tles) == 2:
			return tles
		else:
			print('Something looks off. Here is the most up to date TLE on Celestrak.')
			return self.latest_TLE