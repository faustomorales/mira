import torch
import numpy as np
from mira.detectors import common

examples = list(
    map(
        torch.tensor,
        [
            np.zeros((3, 256, 128)).astype("uint8") + 255,
            np.zeros((3, 128, 256)).astype("uint8") + 255,
            np.zeros((3, 32, 56)).astype("uint8") + 255,
        ],
    )
)


def test_resize_pad():
    resized, scales, sizes = common.resize(
        x=examples, resize_method="pad", height=256, width=256
    )
    assert resized.shape[2:] == (256, 256)
    assert (scales == 1).all()
    for (ylim, xlim), current in zip(sizes, resized):
        assert (current[:, :ylim, :xlim] > 127).all()
        assert (
            np.concatenate(
                [
                    current[:, ylim:].numpy().flatten(),
                    current[:, :, xlim:].numpy().flatten(),
                ]
            ).max()
            < 127
        )


def test_resize_pad_to_multiple():
    resized, scales, sizes = common.resize(
        x=examples, resize_method="pad_to_multiple", base=512
    )
    assert resized.shape[2:] == (512, 512)
    assert (scales == 1).all()
    for (ylim, xlim), current in zip(sizes, resized):
        assert (current[:, :ylim, :xlim] > 127).all()
        assert (
            np.concatenate(
                [
                    current[:, ylim:].numpy().flatten(),
                    current[:, :, xlim:].numpy().flatten(),
                ]
            ).max()
            < 127
        )


def test_resize_fit():
    resized, scales, sizes = common.resize(
        x=examples, resize_method="fit", height=128, width=128
    )
    assert resized.shape[2:] == (128, 128)
    np.testing.assert_allclose(
        scales.numpy()[:, 0], np.array([128 / 256, 128 / 256, 128 / 56])
    )
    np.testing.assert_allclose(
        sizes, np.array([[128, 64], [64, 128], [round(32 * 128 / 56), 128]])
    )
    for (ylim, xlim), current in zip(sizes, resized):
        assert (current[:, :ylim, :xlim] > 127).all()
        assert (
            np.concatenate(
                [
                    current[:, ylim:].numpy().flatten(),
                    current[:, :, xlim:].numpy().flatten(),
                ]
            ).max()
            < 127
        )
