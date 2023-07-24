import datetime

import simulator.filters
from simulator.cloud import Cloud
from simulator.types import Image, Region


def test_evaluate():
    now = datetime.datetime.now()
    sample_image = simulator.types.Image(size=0, region=Region(0, 0, 0, 0), time=now)
    sample_image.set_score(1)
    sample_filter = simulator.filters.IdentityFilter()
    sample_pipeline = simulator.filters.ImageProcessingPipeline([[sample_filter]])
    cloud = Cloud(now, sample_pipeline)
    cloud.upload_images([sample_image])
    assert cloud.evaluate() == 0
    cloud.update(1)
    cloud.upload_images([sample_image])
    assert cloud.evaluate() == .5
