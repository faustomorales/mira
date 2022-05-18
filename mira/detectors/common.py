# pylint: disable=unexpected-keyword-arg
import typing
import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as tvtf
import typing_extensions as tx

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


def torchvision_serve_inference(
    model, x: typing.List[torch.Tensor], resize_config: ResizeConfig
):
    """A convenience function for performing inference using torchvision
    object detectors in a common way with different resize methods."""
    resized, scales, sizes = resize(x, resize_config=resize_config)
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


class NonResizingCRNNTransform(
    torchvision.models.detection.transform.GeneralizedRCNNTransform
):
    """A subclass of torchvision.models.detection.transform that *doesn't*
    resize images (since we usually handle that upstream)."""

    def resize(self, image, target=None):
        return image, target


def convert_rcnn_transform(
    transform: torchvision.models.detection.transform.GeneralizedRCNNTransform,
):
    """Convert RCNN transform to prevent in-model resizing."""
    return NonResizingCRNNTransform(
        min_size=transform.min_size,
        max_size=None,
        image_mean=transform.image_mean,
        image_std=transform.image_std,
        size_divisible=transform.size_divisible,
        fixed_size=None,
    )


def get_torchvision_anchor_boxes(
    model: torch.nn.Module, device: torch.device, height: int, width: int
):
    """Get the anchor boxes for a torchvision model."""
    image_list = torchvision.models.detection.image_list.ImageList(
        tensors=torch.tensor(
            np.random.randn(1, height, width, 3).transpose(0, 3, 1, 2),
            dtype=torch.float32,
            device=device,
        ),
        image_sizes=[(height, width)],
    )
    feature_maps = model.backbone(image_list.tensors)  # type: ignore
    assert len(feature_maps) == len(
        model.anchor_generator.sizes  # type: ignore
    ), f"Number of feature maps ({len(feature_maps)}) does not match number of anchor sizes ({len(model.anchor_generator.sizes)}). This model is misconfigured."  # type: ignore
    return np.concatenate(
        [
            a.cpu()
            for a in model.anchor_generator(  # type: ignore
                image_list=image_list, feature_maps=list(feature_maps.values())
            )
        ]
    )
