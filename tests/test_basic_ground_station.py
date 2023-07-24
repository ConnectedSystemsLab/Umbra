import datetime

from simulator.groundstation import BasicGroundStation
from simulator.types import Image, Region


def test_image_ordering():
    start_time = datetime.datetime.now()
    gs = BasicGroundStation(start_time, [0, 0, 0], 10000)
    images = [
        Image(1000, Region(0, 0, 0, 0), start_time, id='1'),
        Image(1000, Region(0, 0, 0, 0), start_time, id='2'),
        Image(1000, Region(0, 0, 0, 0), start_time, id='3'),
    ]
    gs.downlink_images(images)
    gs.update(10000)
    del images
    img_queue = gs.get_upload_images()
    assert img_queue[0].id == '1'
    assert img_queue[1].id == '2'
    assert img_queue[2].id == '3'


def test_timekeeping():
    start_time = datetime.datetime.now()
    gs = BasicGroundStation(start_time, [0, 0, 0], 10000)
    gs.update(.002)
    assert (gs.timer - start_time).total_seconds() == .002


def test_bandwidth_control():
    start_time = datetime.datetime.now()
    gs = BasicGroundStation(start_time, [0, 0, 0], 1000)
    images = [
        Image(1000, Region(0, 0, 0, 0), start_time, id='1'),
        Image(1000, Region(0, 0, 0, 0), start_time, id='2'),
        Image(1000, Region(0, 0, 0, 0), start_time, id='3'),
    ]
    gs.downlink_images(images)
    gs.update(.5)
    iq = gs.get_upload_images()
    assert len(iq) == 0
    gs.update(.501)
    iq = gs.get_upload_images()
    assert len(iq) == 1
    gs.update(2.01)
    iq = gs.get_upload_images()
    assert len(iq) == 2
