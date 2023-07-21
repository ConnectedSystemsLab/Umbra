import os
import sys
from abc import abstractmethod, ABC

from datetime import datetime
from typing import List

sys.path.insert(0, os.getcwd() + "/../")
from simulator.types import Image

import operator


class AbstractSatellite(ABC):
	def __init__(self, start_time: datetime, id: str, images: List[Image]):
		self.start_time = start_time
		self.id = id
		if len(images) != 0:
			self.images = images
			self.images.sort(key=operator.attrgetter('time'))
		else:
			self.images = []

	@abstractmethod
	def has_image(self) -> bool:
		pass

	@abstractmethod
	def update(self, time: float):
		pass

	@abstractmethod
	def get_cache_size(self):
		pass

	@abstractmethod
	def set_transmit(self, bandwidth: float):
		pass

	@abstractmethod
	def get_position(self) -> List[float]:  # [lat,lng,alt]
		pass

	@abstractmethod
	def get_downlink_images(self) -> List[Image]:
		pass

	@abstractmethod
	def is_finished(self) -> bool:
		pass

	@abstractmethod
	def get_tle(self) -> List[str]:
		pass

	@abstractmethod
	def read_file(self):
		pass
