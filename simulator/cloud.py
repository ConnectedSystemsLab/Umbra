from copy import deepcopy

from datetime import datetime, timedelta
from typing import List, Tuple

from simulator.filters import ImageProcessingPipeline
from simulator.types import Image


class Cloud:
    def __init__(self, start_time: datetime, evaluator: ImageProcessingPipeline):
        self.evaluator = evaluator
        self.pipeline = ImageProcessingPipeline
        self.image_list: List[Tuple[Image, datetime]] = []
        self.timer = start_time

    def upload_images(self, images: List[Image]):
        for image in images:
            self.image_list.append((deepcopy(image), self.timer))

    def update(self, time: float):
        self.timer += timedelta(seconds=time)

    def evaluate(self, metric='score_weighted_latency'):
        score = 0
        if metric == 'latency_discounted_score':
            for img, upload_time in self.image_list:
                single_score, _ = self.evaluator.filter(img)
                score += single_score / (upload_time - img.time).total_seconds()
        elif metric == 'score_weighted_latency':
            weight = 0
            sum = 0
            for img, upload_time in self.image_list:
                single_score, _ = self.evaluator.filter(img)
                weight += single_score
                sum += single_score * (upload_time - img.time).total_seconds()
            score = sum / weight
        return score
