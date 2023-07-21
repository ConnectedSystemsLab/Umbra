from abc import ABC, abstractmethod
from copy import deepcopy

from shapely.geometry import Polygon
from typing import List, Dict, Union

from .types import Image, Region


class Filter(ABC):
    def __init__(self, user=None):
        self.user = user

    @abstractmethod
    def filter(self, image: Image) -> List[int]:  # returns: [score, time]
        pass


class MetadataFilter(Filter, ABC):
    pass


class ImageFilter(Filter, ABC):
    pass


class GeographicFilter(MetadataFilter):
    def filter(self, image: Image) -> List[float]:
        if self.region.overlaps(image.region):
            return [1, 0]
        return [0, 0]

    def __init__(self, user, region: Union[Region, Polygon]):
        super().__init__(user)
        self.region = region


class SavedFilter(Filter):
    def filter(self, image: Image) -> List[float]:
        return self.img_table[image.id]

    def __init__(self, img_table: Dict[str, List[float]]):
        super().__init__()
        self.img_table = img_table


class ImageProcessingPipeline:
    def __init__(self, pipelines: List[List[Filter]], in_queue_hook=None):
        self.pipelines = pipelines
        self.in_queue = in_queue_hook
        self.task_remaining_time = 0
        self.out_queue = []
        self.timer = 0

    def set_in_queue(self, in_queue_hook: List[Image]):
        self.in_queue = in_queue_hook

    def abort(self):
        self.task_remaining_time = 0

    def update(self, time):
        if self.task_remaining_time == 0 and not len(self.in_queue) == 0:
            img_score, self.task_remaining_time = self.filter(self.in_queue[0])
            self.in_queue[0].set_score(-img_score)  # Setting to negative because heapq is a min heap
        self.timer += time
        while time >= self.task_remaining_time:
            time -= self.task_remaining_time
            self.out_queue.append(self.in_queue.pop(0))
            if len(self.in_queue) > 0:
                img_score, self.task_remaining_time = self.filter(self.in_queue[0])
                self.in_queue[0].set_score(-img_score)
            else:
                self.task_remaining_time = time  # So that it will be 0 after the function run
                break
        self.task_remaining_time -= time

    def __put_image__(self):  # TODO write this: take the first image in the input queue, put it in the output queue
        pass

    def get_output_images(self):
        result = deepcopy(self.out_queue)
        self.out_queue = []
        return result

    def filter(self, img: Image):
        total_score = 0
        total_time = 0
        for pipeline in self.pipelines:
            temp_score = 1
            for _filter in pipeline:
                score, time = _filter.filter(img)
                temp_score *= score
                total_time += time
            total_score += temp_score
        return total_score, total_time


class IdentityFilter(Filter):
    def filter(self, image: Image) -> List[int]:
        return [1, 0]
