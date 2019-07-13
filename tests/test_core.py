"""
Test core functions.
"""

import tempfile
import os

from mira.core import Image


def test_blank_and_properties():
    """Make sure creating new images works."""
    image = Image.new(width=200, height=100, channels=2, cval=125)
    assert image.height == 100
    assert image.width == 200
    assert image.channels == 2
    assert (image[:, :, :] == 125).all()


def test_file_read():
    """Make sure reading files works."""
    image = Image.new(width=200, height=100, channels=3, cval=125)
    image[40:60, 40:60, 0] = 0
    with tempfile.TemporaryDirectory() as tempdir:
        fpath = os.path.join(tempdir, 'test.jpg')
        image.save(fpath)
        image = Image.read(fpath)

        def consistency_check(img):
            assert img.width == 200
            assert img.height == 100
            assert img.channels == 3
            assert (img[45:55, 45:55, 0] <= 5).all()
            assert (img[45:55, 45:55, 1:] > 100).all()

        consistency_check(image)
        with open(fpath, 'rb') as buffer:
            image = Image.read(buffer)
        consistency_check(image)

        image = Image.read(image.buffer('.jpeg'))
        consistency_check(image)
