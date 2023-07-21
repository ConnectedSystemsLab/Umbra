from abc import abstractmethod, ABC
from typing import List

from .types import Image


class AbstractGroundstation(ABC):

	@abstractmethod
	def update(self, time: float):
		pass

	@abstractmethod
	def get_position(self):
		pass

	@abstractmethod
	def downlink_images(self, img: List[Image]):
		pass

	@abstractmethod
	def get_upload_images(self) -> List[Image]:
		pass

	@abstractmethod
	def is_finished(self) -> bool:
		pass

	@abstractmethod
	def get_id(self) -> str:
		pass

	@abstractmethod
	def get_bandwidth(self) -> float:
		pass

	@abstractmethod
	def get_queue_length(self):
		pass
