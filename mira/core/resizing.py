import typing

import cv2
import numpy as np
import typing_extensions as tx

try:
    import torch
except ImportError:
    torch = None  # type: ignore
try:
    import torchvision.transforms.functional as tvtf
except ImportError:
    tvtf = None  # type: ignore

FixedSizeConfig = tx.TypedDict(
    "FixedSizeConfig",
    {"method": tx.Literal["fit", "pad", "force"], "width": int, "height": int},
)
VariableSizeConfig = tx.TypedDict(
    "VariableSizeConfig", {"method": tx.Literal["pad_to_multiple"], "base": int}
)
ResizeConfig = typing.Union[
    FixedSizeConfig,
    VariableSizeConfig,
]

ArrayType = typing.TypeVar("ArrayType", torch.Tensor, np.ndarray)


def fit(
    image: ArrayType, height: int, width: int, force: bool
) -> typing.Tuple[ArrayType, typing.Tuple[float, float], typing.Tuple[int, int]]:
    """Fit an image to a specific size, padding where necessary to maintain
    aspect ratio.

    Args:
        image: A tensor with shape (C, H, W) or a numpy array with shape (H, W, C)
    """
    use_torch_ops = isinstance(image, torch.Tensor)
    input_height, input_width = image.shape[1:] if use_torch_ops else image.shape[:2]
    if width == input_width and height == input_height:
        return image, (1.0, 1.0), (height, width)
    if force:
        return (
            tvtf.resize(image, size=[height, width])
            if use_torch_ops
            else cv2.resize(image, dsize=(width, height)),
            (height / input_height, width / input_width),
            (height, width),
        )
    scale = min(width / input_width, height / input_height)
    target_height = int(scale * input_height)
    target_width = int(scale * input_width)
    pad_y = height - target_height
    pad_x = width - target_width
    resized = (
        tvtf.resize(image, size=[target_height, target_width])
        if use_torch_ops
        else cv2.resize(image, (target_width, target_height))
    )
    if pad_y > 0 or pad_x > 0:
        padded = (
            torch.nn.functional.pad(resized, (0, pad_x, 0, pad_y))
            if use_torch_ops
            else np.pad(resized, ((0, pad_y), (0, pad_x), (0, 0)))
        )
    else:
        padded = resized
    return padded, (scale, scale), (target_height, target_width)


def resize(
    x: typing.List[ArrayType], resize_config: ResizeConfig
) -> typing.Tuple[ArrayType, ArrayType, ArrayType]:
    """Resize a list of images using a specified method.

    Args:
        x: A list of tensors of shape (C, H, W) or a tensor of
           shape (N, C, H, W) or a list of ndarrays of shape (H, W, C)
           or an ndarray of shape (N, H, W, C).
        resize_config: A resize config object.
    """
    assert (
        not isinstance(x, list) or len(x) > 0
    ), "When providing a list, it must not be empty."
    assert resize_config["method"] in [
        "fit",
        "pad",
        "none",
        "force",
        "pad_to_multiple",
    ], f"Unknown resize method {resize_config['method']}."
    use_torch_ops = isinstance(x, torch.Tensor) or (
        isinstance(x, list) and isinstance(x[0], torch.Tensor)
    )
    width, height, base = typing.cast(
        typing.List[typing.Optional[int]],
        [resize_config.get(k) for k in ["width", "height", "base"]],
    )
    if resize_config["method"] in ["fit", "force"]:
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using fit or force."
        resized_list, scale_list, size_list = zip(
            *[
                fit(
                    image,
                    height=height,
                    width=width,
                    force=resize_config["method"] == "force",
                )
                for image in x
            ]
        )
        return (
            (  # type: ignore
                torch.cat([r.unsqueeze(0) for r in resized_list]),
                torch.tensor(scale_list),
                torch.tensor(size_list),
            )
            if use_torch_ops
            else (
                np.concatenate([r[np.newaxis] for r in resized_list]),
                np.array(scale_list),
                np.array(size_list),
            )
        )
    img_dimensions = np.array(
        [i.shape[1:3] if use_torch_ops else i.shape[:2] for i in x]
    )
    scales = (
        torch.tensor(np.ones_like(img_dimensions))
        if use_torch_ops
        else np.ones_like(img_dimensions)
    )
    sizes = torch.tensor(img_dimensions) if use_torch_ops else np.array(img_dimensions)
    if resize_config["method"] == "pad":
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using pad."
        pad_dimensions = np.array([[height, width]]) - img_dimensions
    if resize_config["method"] == "pad_to_multiple":
        assert base is not None, "pad_to_multiple requires a base to be provided."
        pad_dimensions = (
            (np.ceil(img_dimensions.max(axis=0) / base) * base)
            .clip(base)
            .astype("int32")
        ) - img_dimensions
    if resize_config["method"] == "none":
        # pylint: disable=unexpected-keyword-arg
        pad_dimensions = img_dimensions.max(axis=0, keepdims=True) - img_dimensions
    assert (
        pad_dimensions >= 0
    ).all(), "Input image is larger than target size, but method is 'pad.'"
    padded = (
        torch.cat(
            [
                torch.nn.functional.pad(i, (0, pad_x, 0, pad_y)).unsqueeze(0)  # type: ignore
                for i, (pad_y, pad_x) in zip(x, pad_dimensions)
            ]
        )
        if use_torch_ops
        else np.concatenate(
            [
                np.pad(i, ((0, pad_y), (0, pad_x), (0, 0)))[np.newaxis]
                for i, (pad_y, pad_x) in zip(x, pad_dimensions)
            ],
            axis=0,
        )
    )
    return padded, scales, sizes  # type: ignore
