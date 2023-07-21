import fcntl
import os.path
import pickle as pkl
import requests
import os

class WeatherService:
	def __init__(self, fn):
		self.fn = fn
		self.key = os.environ['DARKSKY_KEY']

	def get_weather_info(self, lat, long, time):
		get_info = str(lat) + ',' + str(long) + ',' + str(time)
		get_info_path = os.path.join(self.fn, get_info)
		weather = None
		if os.path.exists(get_info_path):
			with open(get_info_path, 'rb') as f:
				try:
					fcntl.flock(f, fcntl.LOCK_SH)
					weather = pkl.load(f)
				except EOFError:
					pass
				finally:
					fcntl.flock(f, fcntl.LOCK_UN)
		if weather is None:
			print(get_info)
			self.resp = requests.get(
				f'https://api.darksky.net/forecast/{self.key}' + '/' + get_info + '/?units=si')
			print(self.resp)
			assert self.resp.ok
			weather = self.resp.json()
			with open(get_info_path, 'wb') as f:
				fcntl.flock(f, fcntl.LOCK_EX)
				pkl.dump(weather, f)
				fcntl.flock(f, fcntl.LOCK_UN)
		return weather

	def commit(self):
		pass
