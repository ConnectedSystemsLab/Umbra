from datetime import datetime
from numpy import ndarray
from shapely.geometry import Polygon
from typing import Union


class Region:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.max_y = max_y
        self.max_x = max_x
        self.min_y = min_y
        self.min_x = min_x

    def contain(self, coordinate):
        x, y = coordinate
        if self.min_x < x < self.max_x and self.min_y < y < self.max_y:
            return True
        return False

    def contains(self, region):
        if max(self.min_x, region.min_x) < min(self.max_x, region.max_x):
            if max(self.min_y, region.min_y) < min(self.max_y, region.max_y):
                return True
        return False


class Image:
    def __init__(self, size: float, region: Union[Region, Polygon], time: datetime, mask: ndarray = None, id=None):
        self.mask = mask
        self.id = id
        self.time = time
        self.region = region
        self.size = size
        self.score = 0

    def set_score(self, value):
        self.score = value

    @staticmethod
    def from_dict(data):
        min_x, min_y, max_x, max_y = data['region']
        region = Region(min_x, min_y, max_x, max_y)
        return Image(
            size=data['size'],
            id=data['id'],
            time=data['time'],
            region=region
        )

    # To implement custom comparator (on the score) for the priority queue in the detector
    def __lt__(self, obj):
        """self < obj."""
        return self.score < obj.score

    def __le__(self, obj):
        """self <= obj."""
        return self.score <= obj.score

    def __eq__(self, obj):
        """self == obj."""
        return self.score == obj.score

    def __ne__(self, obj):
        """self != obj."""
        return not self.score == obj.score

    def __gt__(self, obj):
        """self > obj."""
        return self.score > obj.score

    def __ge__(self, obj):
        """self >= obj."""
        return self.score >= obj.score
