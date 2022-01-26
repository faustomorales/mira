# pylint: disable=unexpected-keyword-arg
import typing
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as tvtf


class LastLevelNoop(torchvision.ops.feature_pyramid_network.ExtraFPNBlock):
    """
    A noop extra FPN block. Use this to force a noop for functions
    that automatically insert an FPN block when you set
    extra_blocks to None.
    """

    # "results" refers to feature pyramid outputs
    # "x" refers to convolutional layer outputs.
    def forward(self, results, x, names):  # pylint: disable=unused-argument
        return results, names


def fit(
    image: torch.Tensor, height: int, width: int
) -> typing.Tuple[torch.Tensor, float, typing.Tuple[int, int]]:
    """Fit an image to a specific size, padding where necessary to maintain
    aspect ratio.

    Args:
        image: A tensor with shape (C, H, W)
    """
    input_height, input_width = image.shape[1:]
    if width == input_width and height == input_height:
        return image, 1.0, (height, width)
    scale = min(width / input_width, height / input_height)
    target_height = int(scale * input_height)
    target_width = int(scale * input_width)
    pad_y = height - target_height
    pad_x = width - target_width
    resized = tvtf.resize(image, size=[target_height, target_width])
    if pad_y > 0 or pad_x > 0:
        padded = torch.nn.functional.pad(resized, (0, pad_x, 0, pad_y))
    else:
        padded = resized
    return padded, scale, (target_height, target_width)


def resize(
    x: typing.Union[typing.List[torch.Tensor], torch.Tensor],
    resize_method: typing.Optional[str],
    height: int = None,
    width: int = None,
    base: int = None,
):
    """Resize a list of images using a specified method.

    Args:
        x: A list of arrays of shape (C, H, W) or an array of
           shape (N, C, H, W)
        resize_method: One of fit (distort image to fit), pad (pad to
            target height and width) or pad_to_multiple (pad to multiple
            of some given base integer).
    """
    if resize_method is not None and resize_method == "fit":
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using fit."
        resized_list, scale_list, size_list = zip(
            *[fit(image, height=height, width=width) for image in x]
        )
        return (
            torch.cat([r.unsqueeze(0) for r in resized_list]),
            torch.tensor(scale_list).unsqueeze(1).repeat((1, 2)),
            torch.tensor(size_list),
        )
    img_dimensions = np.array([i.shape[1:3] for i in x])
    scales = torch.tensor(np.ones_like(img_dimensions))
    sizes = torch.tensor(img_dimensions)
    pad_dimensions = None
    if resize_method is not None and resize_method == "pad":
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using pad."
        pad_dimensions = np.array([[height, width]]) - img_dimensions
    if resize_method is not None and resize_method.startswith("pad_to_multiple"):
        assert base is not None, "pad_to_multiple requires a base to be provided."
        pad_dimensions = (
            (np.ceil(img_dimensions.max(axis=0) / base) * base)
            .clip(base)
            .astype("int32")
        ) - img_dimensions
    if resize_method is None:
        pad_dimensions = img_dimensions.max(axis=0, keepdims=True) - img_dimensions
    if pad_dimensions is None:
        # None of the padding methods applied.
        raise NotImplementedError("Unknown resize method:", resize_method)
    assert (
        pad_dimensions >= 0
    ).all(), "Input image is larger than target size, but resize_method is 'pad.'"
    padded = torch.cat(
        [
            torch.nn.functional.pad(i, (0, pad_x, 0, pad_y)).unsqueeze(0)
            for i, (pad_y, pad_x) in zip(x, pad_dimensions)
        ]
    )
    return padded, scales, sizes


def torchvision_serve_inference(
    model,
    x: typing.Union[typing.List[torch.Tensor], torch.Tensor],
    resize_method: typing.Optional[str],
    height: int = None,
    width: int = None,
    base: int = None,
):
    """A convenience function for performing inference using torchvision
    object detectors in a common way with different resize methods."""
    resized, scales, sizes = resize(
        x,
        resize_method=resize_method,
        height=height,
        width=width,
        base=base,
    )
    # Go from [sy, sx] to [sx, sy, sx, sy]
    scales = scales[:, [1, 0]].repeat((1, 2))
    sizes = sizes[:, [1, 0]].repeat((1, 2))
    if (
        model.input_shape[2],
        model.input_shape[0],
        model.input_shape[1],
    ) != resized.shape[1:]:
        model.set_input_shape(height=resized.shape[2], width=resized.shape[3])
    detections = [
        {k: v.detach().cpu() for k, v in d.items()} for d in model.model(resized)
    ]
    clipped = [
        group["boxes"][:, :4].min(group_size.unsqueeze(0))
        for group, group_size in zip(detections, sizes)
    ]
    has_area = [((c[:, 3] - c[:, 1]) * (c[:, 2] - c[:, 0])) > 0 for c in clipped]
    return [
        {
            "boxes": group["boxes"][box_has_area] * (1 / scaler),
            "labels": group["labels"][box_has_area],
            "scores": group["scores"][box_has_area],
        }
        for group, box_has_area, scaler in zip(detections, has_area, scales)
    ]
