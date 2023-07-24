import datetime

from simulator.filters import ImageProcessingPipeline, Filter, IdentityFilter
from simulator.groundstation import StandardGroundstation
from simulator.types import Image, Region


class TestFilter(Filter):
    def __init__(self, id_mapping):
        super().__init__([[]])
        self.id_mapping = id_mapping

    def filter(self, img):
        return self.id_mapping[img.id]


def test_image_ordering():
    pass
    node0 = TestFilter({
        0: [.1, .2],
        1: [.2, .2],
        2: [.3, .2]
    })
    node1 = TestFilter({
        0: [.3, .2],
        1: [.2, .2],
        2: [.2, .2]
    })
    pipeline = ImageProcessingPipeline([[node0, node1]])
    pre_pipeline = ImageProcessingPipeline([[IdentityFilter()]])
    gs = StandardGroundstation(datetime.datetime.now(), [0, 0, 0], pre_pipeline, pipeline, 1000)
    images = [
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=0),
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=1),
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=2),
    ]
    gs.downlink_images(images)
    gs.update(10)
    img_list = gs.get_upload_images()
    assert img_list[0].id == 2
    assert img_list[1].id == 1
    assert img_list[2].id == 0


def test_timekeeping():
    node0 = TestFilter({
        0: [.1, .2],
        1: [.2, .2],
        2: [.3, .2]
    })
    node1 = TestFilter({
        0: [.3, .3],
        1: [.2, .2],
        2: [.2, .4]
    })
    start_time = datetime.datetime.now()
    gs = StandardGroundstation(start_time, [0, 0, 0], ImageProcessingPipeline([[node0]]),
                               ImageProcessingPipeline([[node0, node1]]), 1)
    gs.update(.002)
    assert (gs.timer - start_time).total_seconds() == .002
    assert gs.detector.timer == .002
    images = [
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=0),
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=1),
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=2),
    ]
    gs.downlink_images(images)
    gs.update(.201)
    assert len(gs.img_out_queue) == 0
    gs.update(.401)
    assert len(gs.img_out_queue) == 1
    gs.update(.401)
    assert len(gs.img_out_queue) == 2


def test_bandwidth_control():
    node0 = TestFilter({
        0: [.1, .0],
        1: [.2, .0],
        2: [.3, .0]
    })
    node1 = TestFilter({
        0: [.3, .0],
        1: [.2, .0],
        2: [.2, .0]
    })
    start_time = datetime.datetime.now()
    gs = StandardGroundstation(start_time, [0, 0, 0], ImageProcessingPipeline([[node0]]),
                               ImageProcessingPipeline([[node0, node1]]), 1)
    images = [
        Image(3, Region(0, 0, 0, 0), datetime.datetime.now(), id=0),
        Image(2, Region(0, 0, 0, 0), datetime.datetime.now(), id=1),
        Image(1, Region(0, 0, 0, 0), datetime.datetime.now(), id=2),
    ]
    gs.downlink_images(images)
    gs.update(1.001)
    upload_list = gs.get_upload_images()
    assert len(upload_list) == 1
    gs.update(1.001)
    upload_list = gs.get_upload_images()
    assert len(upload_list) == 0
    gs.update(4.001)
    upload_list = gs.get_upload_images()
    assert len(upload_list) == 2
