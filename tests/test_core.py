"""
Test core functions.
"""

import tempfile
import os
import io

import numpy as np

from mira import core, datasets


def test_blank_and_properties():
    """Make sure creating new images works."""
    image = core.utils.get_blank_image(width=200, height=100, n_channels=2, cval=125)
    assert image.shape == (100, 200, 2)
    assert (image[:, :, :] == 125).all()


def test_file_read():
    """Make sure reading files works."""
    image = core.utils.get_blank_image(width=200, height=100, n_channels=3, cval=125)
    image[40:60, 40:60, 0] = 0
    with tempfile.TemporaryDirectory() as tempdir:
        fpath = os.path.join(tempdir, "test.png")
        core.utils.save(image, fpath)
        np.testing.assert_allclose(core.utils.read(fpath), image)
        with open(fpath, "rb") as buffer:
            np.testing.assert_allclose(core.utils.read(buffer), image)
        with io.BytesIO() as buffer:
            core.utils.save(image, buffer, ".png")
            buffer.seek(0)
            np.testing.assert_allclose(core.utils.read(buffer), image)
