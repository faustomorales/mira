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


SideOptions = tx.Literal["longest", "shortest"]
FixedSizeConfig = tx.TypedDict(
    "FixedSizeConfig",
    {
        "method": tx.Literal["fit", "pad", "force"],
        "width": int,
        "height": int,
        "cval": int,
    },
)
VariableSizeConfig = tx.TypedDict(
    "VariableSizeConfig",
    {
        "method": tx.Literal["pad_to_multiple"],
        "base": int,
        "max": typing.Optional[int],
        "cval": int,
    },
)
AspectPreservingConfig = tx.TypedDict(
    "AspectPreservingConfig",
    {
        "method": tx.Literal["fit_side"],
        "target": int,
        "side": SideOptions,
        "upsample": bool,
    },
)
ResizeConfig = typing.Union[FixedSizeConfig, VariableSizeConfig, AspectPreservingConfig]

ArrayType = typing.TypeVar("ArrayType", "torch.Tensor", np.ndarray)


def call_resize(
    image: typing.Any,
    target_height: int,
    target_width: int,
    interpolation: typing.Any = None,
):
    """Underlying resize call which uses cv2 or torch depending on argument."""
    use_torch_ops = torch and isinstance(image, torch.Tensor)
    return (
        tvtf.resize(
            image,
            size=[target_height, target_width],
            interpolation=interpolation or tvtf.InterpolationMode.BILINEAR,
        )
        if use_torch_ops
        else cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation or cv2.INTER_LINEAR,
        )
    )


def fit_side(
    image: ArrayType,
    target_length: int,
    side: SideOptions,
    upsample: bool,
    interpolation: typing.Any = None,
) -> typing.Tuple[ArrayType, typing.Tuple[float, float], typing.Tuple[int, int]]:
    """Fit an image such that the longest or shortest side matches some specific target length. Not
    supported for batch resizing operations."""
    use_torch_ops = torch and isinstance(image, torch.Tensor)
    input_height, input_width = image.shape[1:] if use_torch_ops else image.shape[:2]
    func_lookup = {"longest": max, "shortest": min}
    reference_length = func_lookup[side](input_height, input_width)
    scale = target_length / reference_length
    if not upsample:
        scale = min(scale, 1.0)
    if scale == 1.0:
        return image, (1.0, 1.0), (input_height, input_width)
    target_width, target_height = [
        target_length if reference_length == isize else int(round(scale * isize))
        for isize in [input_width, input_height]
    ]
    resized = call_resize(
        image=image,
        target_height=target_height,
        target_width=target_width,
        interpolation=interpolation,
    )
    return resized, (scale, scale), (target_height, target_width)


def fit(
    image: ArrayType,
    height: int,
    width: int,
    force: bool,
    cval=0,
    interpolation: typing.Any = None,
) -> typing.Tuple[ArrayType, typing.Tuple[float, float], typing.Tuple[int, int]]:
    """Fit an image to a specific size, padding where necessary to maintain
    aspect ratio.

    Args:
        image: A tensor with shape (C, H, W) or a numpy array with shape (H, W, C)
    """
    use_torch_ops = torch and isinstance(image, torch.Tensor)
    input_height, input_width = image.shape[1:] if use_torch_ops else image.shape[:2]
    if width == input_width and height == input_height:
        return image, (1.0, 1.0), (height, width)
    if force:
        return (
            call_resize(
                image=image,
                target_height=height,
                target_width=width,
                interpolation=interpolation,
            ),
            (height / input_height, width / input_width),
            (height, width),
        )
    scale = min(width / input_width, height / input_height)
    target_height = int(scale * input_height)
    target_width = int(scale * input_width)
    pad_y = height - target_height
    pad_x = width - target_width
    resized = call_resize(
        image=image,
        target_height=target_height,
        target_width=target_width,
        interpolation=interpolation,
    )
    if pad_y > 0 or pad_x > 0:
        padded = (
            torch.nn.functional.pad(
                resized, (0, pad_x, 0, pad_y), mode="constant", value=cval
            )
            if use_torch_ops
            else np.pad(
                resized,
                ((0, pad_y), (0, pad_x))
                + (((0, 0),) if len(resized.shape) == 3 else tuple()),
                mode="constant",
                constant_values=cval,
            )
        )
    else:
        padded = resized
    return typing.cast(ArrayType, padded), (scale, scale), (target_height, target_width)


def check_resize_method(method: str):
    """Verify that a resize method is okay."""
    assert method in [
        "fit",
        "pad",
        "force",
        "none",
        "pad_to_multiple",
        "fit_side",
    ], f"Unknown resize method {method}."


def compute_resize_dimensions(shapes: np.ndarray, resize_config: ResizeConfig):
    """Compute the expected output size for a series of sizes (avoids having
    to actually "do" the resizing. Note that this function follows the y, x
    convention."""
    check_resize_method(resize_config["method"])
    max_size = shapes.max(axis=0)
    if resize_config["method"] in ("fit", "pad", "force"):
        resize_config = typing.cast(FixedSizeConfig, resize_config)
        dimensions = np.array([resize_config["height"], resize_config["width"]])
    elif resize_config["method"] == "pad_to_multiple":
        base = resize_config["base"]
        dimensions = (np.ceil(max_size / base) * base).round().astype("int32")
    elif resize_config["method"] == "none":
        dimensions = max_size
    else:
        raise ValueError("Failed to compute dimensions.")
    if resize_config["method"] == "fit":
        scalesout = (
            (dimensions / shapes).min(axis=1)[:, np.newaxis].repeat(repeats=2, axis=1)
        )
        sizesout = (scalesout * shapes).round()
    elif resize_config["method"] == "pad":
        scalesout = np.ones((len(shapes), 2))
        sizesout = shapes
        assert (shapes <= dimensions).all(), "Cannot pad images to this size."
    elif resize_config["method"] == "pad_to_multiple":
        pad_max = resize_config.get("max")
        if pad_max is not None:
            assert pad_max % base == 0, "max padding size must be multiple of base."
            dimensions = dimensions.clip(0, pad_max)
        scalesout = np.ones((len(shapes), 2))
        downsampled = (dimensions < shapes).any(axis=1)
        scalesout[downsampled] = (
            (dimensions / shapes)[downsampled]
            .min(axis=1)[:, np.newaxis]
            .repeat(repeats=2, axis=1)
        )
        sizesout = (scalesout * shapes).round()
    elif resize_config["method"] == "force":
        scalesout = dimensions / shapes
        sizesout = dimensions.repeat(repeats=len(shapes), axis=0)
    # Follow the convention from resize where we provide dimensions in
    # y, x order.
    return (
        dimensions[np.newaxis, ::-1].repeat(repeats=len(shapes), axis=0),
        scalesout,
        sizesout,
    )


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
    check_resize_method(resize_config["method"])
    use_torch_ops = torch and (
        isinstance(x, torch.Tensor)
        or (isinstance(x, list) and isinstance(x[0], torch.Tensor))
    )
    if resize_config["method"] == "fit_side":
        assert len(x) == 1, "fit_side does not support batches."
        resized, scale, size = fit_side(
            image=x[0],
            target_length=resize_config["target"],
            side=resize_config["side"],
            upsample=resize_config["upsample"],
        )
        return (
            (  # type: ignore
                typing.cast(torch.Tensor, resized).unsqueeze(0),
                torch.tensor([scale]),
                torch.tensor([size]),
            )
            if use_torch_ops
            else (
                resized[np.newaxis],
                np.array([scale]),
                np.array([size]),
            )
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
    raw_shapes = np.array([i.shape[1:3] if use_torch_ops else i.shape[:2] for i in x])
    sizes_arr = raw_shapes.copy()
    scales_arr = np.ones_like(sizes_arr, dtype="float32")
    if resize_config["method"] == "pad":
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using pad."
        pad_dimensions = np.array([[height, width]]) - sizes_arr
        assert (
            pad_dimensions >= 0
        ).all(), "Input image is larger than target size, but method is 'pad.'"
    if resize_config["method"] == "pad_to_multiple":
        assert base is not None, "pad_to_multiple requires a base to be provided."
        pad_max = resize_config.get("max")
        if pad_max is not None:
            assert pad_max % base == 0, "max padding size must be multiple of base."
            sizes_arr = sizes_arr.clip(0, pad_max)
        downsampled = (sizes_arr < raw_shapes).any(axis=1)
        scales_arr[downsampled] = (
            (sizes_arr / raw_shapes)[downsampled]
            .min(axis=1)[:, np.newaxis]
            .repeat(repeats=2, axis=1)
        )
        sizes_arr = (scales_arr * raw_shapes).round().astype("int32")
        pad_dimensions = (
            (np.ceil(sizes_arr.max(axis=0) / base) * base).clip(base).astype("int32")
        ) - raw_shapes
    if resize_config["method"] == "none":
        # pylint: disable=unexpected-keyword-arg
        pad_dimensions = sizes_arr.max(axis=0, keepdims=True) - sizes_arr
    cval = resize_config.get("cval", 0)
    padded = (
        torch.cat(
            [
                (
                    torch.nn.functional.pad(i, (0, pad_x, 0, pad_y), mode="constant", value=cval).unsqueeze(0)  # type: ignore
                    if pad_y >= 0 and pad_x >= 0
                    else fit(
                        typing.cast(torch.Tensor, i),
                        height=raw_height + pad_y,
                        width=raw_width + pad_x,
                        force=False,
                    )[0].unsqueeze(0)
                )
                for i, (pad_y, pad_x), (raw_height, raw_width) in zip(
                    x, pad_dimensions, raw_shapes
                )
            ]
        )
        if use_torch_ops
        else np.concatenate(
            [
                (
                    np.pad(  # type: ignore
                        i,
                        ((0, pad_y), (0, pad_x))
                        + (((0, 0),) if len(i.shape) == 3 else tuple()),
                        mode="constant",
                        constant_values=cval,
                    )[np.newaxis]
                    if pad_y >= 0 and pad_x >= 0
                    else fit(
                        i,
                        height=raw_height + pad_y,
                        width=raw_width + pad_x,
                        force=False,
                    )[0][np.newaxis]
                )
                for i, (pad_y, pad_x), (raw_height, raw_width) in zip(
                    x, pad_dimensions, raw_shapes
                )
            ],
            axis=0,
        )
    )
    scales = torch.tensor(scales_arr) if use_torch_ops else scales_arr
    sizes = torch.tensor(sizes_arr) if use_torch_ops else sizes_arr
    return padded, scales, sizes  # type: ignore
