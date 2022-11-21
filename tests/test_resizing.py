import torch
import pytest
import numpy as np
from mira.core.resizing import resize

base_examples = [
    np.zeros((256, 128, 3)).astype("uint8") + 255,
    np.zeros((128, 256, 3)).astype("uint8") + 255,
    np.zeros((32, 56, 3)).astype("uint8") + 255,
]
typed_examples = [
    (base_examples, False),
    ([torch.tensor(x.transpose(2, 0, 1)) for x in base_examples], True),
]


def verify_results(values, xlim, ylim, tensor_mode):
    assert (
        (values[:, :ylim, :xlim] if tensor_mode else values[:ylim, :xlim]) > 127
    ).all()
    assert (
        np.concatenate(
            [
                (values[:, ylim:].numpy() if tensor_mode else values[ylim:]).flatten(),
                (
                    values[:, :, xlim:].numpy() if tensor_mode else values[:, xlim:]
                ).flatten(),
            ]
        ).max()
        < 127
    )


@pytest.mark.parametrize("examples,tensor_mode", typed_examples)
def test_resize_pad(examples, tensor_mode):
    resized, scales, sizes = resize(
        x=examples,
        resize_config={"method": "pad", "height": 256, "width": 256, "cval": 0},
    )
    assert (resized.shape[2:] if tensor_mode else resized.shape[1:3]) == (256, 256)
    assert (scales == 1).all()
    for (ylim, xlim), current in zip(sizes, resized):
        verify_results(current, xlim=xlim, ylim=ylim, tensor_mode=tensor_mode)


@pytest.mark.parametrize("examples,tensor_mode", typed_examples)
def test_resize_pad_to_multiple(examples, tensor_mode):
    resized, scales, sizes = resize(
        x=examples,
        resize_config={
            "method": "pad_to_multiple",
            "base": 512,
            "max": None,
            "cval": 0,
        },
    )
    assert (resized.shape[2:] if tensor_mode else resized.shape[1:3]) == (512, 512)
    assert (scales == 1).all()
    for (ylim, xlim), current in zip(sizes, resized):
        verify_results(current, xlim=xlim, ylim=ylim, tensor_mode=tensor_mode)


@pytest.mark.parametrize("examples,tensor_mode", typed_examples)
def test_resize_fit(examples, tensor_mode):
    resized, scales, sizes = resize(
        x=examples,
        resize_config={"method": "fit", "height": 128, "width": 128, "cval": 0},
    )
    assert (resized.shape[2:] if tensor_mode else resized.shape[1:3]) == (128, 128)
    np.testing.assert_allclose(scales[:, 0], np.array([128 / 256, 128 / 256, 128 / 56]))
    np.testing.assert_allclose(
        sizes, np.array([[128, 64], [64, 128], [round(32 * 128 / 56), 128]])
    )
    for (ylim, xlim), current in zip(sizes, resized):
        verify_results(current, xlim=xlim, ylim=ylim, tensor_mode=tensor_mode)
