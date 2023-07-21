import heapq
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Union

from simulator.abstract_groundstation import AbstractGroundstation
from simulator.filters import ImageProcessingPipeline
from simulator.types import Image


class BasicGroundStation(AbstractGroundstation):
	def get_queue_length(self):
		return len(self.img_queue)

	def update(self, time: float):
		self.timer += timedelta(seconds=time)
		self.upload_quota += self.get_bandwidth() * time
		self.step += 1

	def get_bandwidth(self) -> float:
		return self.bandwidth if isinstance(self.bandwidth, float) or isinstance(self.bandwidth, int) else \
		self.bandwidth[self.step]

	def set_bandwidth(self, bandwidth: Union[float, List[float]]):
		self.bandwidth = bandwidth

	def get_buffer_size(self) -> float:
		return sum([img.size for img in self.img_queue])

	def get_position(self):
		return self.position

	def downlink_images(self, img: List[Image]):
		self.img_queue.extend(deepcopy(img))

	def get_upload_images(self) -> List[Image]:
		upload_list = []
		while len(self.img_queue) > 0:
			img_top = self.img_queue.pop(0)
			if img_top.size > self.upload_quota:
				self.img_queue = [img_top] + self.img_queue
				break
			self.upload_quota -= img_top.size
			upload_list.append(img_top)
		if self.is_finished():
			self.upload_quota = 0
		return upload_list

	def is_finished(self) -> bool:
		return len(self.img_queue) == 0

	def __init__(self, start_time: datetime, position, bandwidth: Union[float, List[float]], name: str = ""):
		self.img_queue = []
		self.upload_quota = 0
		self.bandwidth = bandwidth
		self.timer = start_time
		self.position = position
		self.name = name
		self.step = 0

	def get_id(self) -> str:
		return self.name


class StandardGroundstation(AbstractGroundstation):
	def get_bandwidth(self) -> float:
		if isinstance(self.bandwidth, float):
			return self.bandwidth
		else:
			return self.bandwidth[self.step]

	def get_queue_length(self):
		return len(self.img_in_queue) + len(self.img_out_queue)

	def is_finished(self) -> bool:
		return len(self.img_in_queue) == 0 and len(self.img_out_queue) == 0

	def get_buffer_size(self) -> float:
		return sum([img.size for img in self.img_in_queue]) + sum([img.size for img in self.img_out_queue])

	def get_upload_images(self) -> List[Image]:
		upload_list = []
		while len(self.img_out_queue) > 0:
			img_top = heapq.heappop(self.img_out_queue)
			if img_top.size > self.upload_quota:
				heapq.heappush(self.img_out_queue, img_top)
				break
			self.upload_quota -= img_top.size
			upload_list.append(img_top)
		while len(self.img_in_queue) > 0:
			img_top = heapq.heappop(self.img_in_queue)
			if img_top.size > self.upload_quota:
				heapq.heappush(self.img_in_queue, img_top)
				break
			self.upload_quota -= img_top.size
			upload_list.append(img_top)
			self.detector.abort()
		return upload_list

	def update(self, time: float):
		self.step += 1
		self.timer += timedelta(seconds=time)
		if not self.is_finished():
			self.upload_quota += self.get_bandwidth() * time
		else:
			self.upload_quota = 0
		self.detector.update(time)
		completed_images = self.detector.get_output_images()
		for item in completed_images:
			heapq.heappush(self.img_out_queue, item)

	def get_position(self):
		return self.position

	def downlink_images(self, imgs: List[Image]):
		for image in imgs:
			score, _ = self.preprocessor.filter(image)
			image.set_score(score)
			heapq.heappush(self.img_in_queue, image)

	def __init__(self, start_time: datetime, position, preprocessor: ImageProcessingPipeline,
	             detector: ImageProcessingPipeline, bandwidth: Union[float, List[float]], name: str = ""):
		self.img_in_queue = []
		self.img_out_queue = []
		self.upload_quota = 0
		self.bandwidth = bandwidth
		self.detector = detector
		self.preprocessor = preprocessor
		self.detector.set_in_queue(self.img_in_queue)
		self.timer = start_time
		self.position = position
		self.name = name
		self.step = 0

	def get_id(self) -> str:
		return self.name
