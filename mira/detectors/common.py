# pylint: disable=unexpected-keyword-arg,unused-argument
import typing
import collections
import timm
import torch
import numpy as np
import torchvision
from .. import core


class LastLevelNoop(torchvision.ops.feature_pyramid_network.ExtraFPNBlock):
    """
    A noop extra FPN block. Use this to force a noop for functions
    that automatically insert an FPN block when you set
    extra_blocks to None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    # args are pyramid outputs, conv outputs, and names
    # We can't use named arguments because the signature for
    # lastlevelp6py differs from that of maxpool which differs
    # from that of the base class.
    def forward(self, *args):  # pylint: disable=unused-argument
        return args[0], args[2]


class LastLevelMaxPool(torchvision.ops.feature_pyramid_network.LastLevelMaxPool):
    """A last level max pool that ignores in_channels / out_channels arguments."""

    def __init__(self, *args, in_channels=None, out_channels=None, **kwargs):
        super().__init__(*args, **kwargs)


EXTRA_BLOCKS_MAP = {
    "lastlevelp6p7": torchvision.ops.feature_pyramid_network.LastLevelP6P7,
    "noop": LastLevelNoop,
    "lastlevelmaxpool": LastLevelMaxPool,
}


class BackboneWithTIMM(torch.nn.Module):
    """An experimental class that operates Like BackboneWithFPN but built using a
    model built using timm.create_model."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        out_channels=None,
        extra_blocks=None,
        **kwargs,
    ):
        super().__init__()
        self.body = timm.create_model(
            model_name=model_name, pretrained=pretrained, **kwargs, features_only=True
        )
        self.out_channels = out_channels or max(self.body.feature_info.channels())
        self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(
            in_channels_list=self.body.feature_info.channels(),
            out_channels=self.out_channels,
        )
        if extra_blocks is not None:
            self.extra_blocks = extra_blocks(
                in_channels=self.out_channels, out_channels=self.out_channels
            )
        else:
            self.extra_blocks = None

    # pylint: disable=missing-function-docstring
    def forward(self, x):
        names = self.body.feature_info.module_name()
        c = self.body(x)
        p = list(self.fpn(collections.OrderedDict(zip(names, c))).values())
        if self.extra_blocks is not None:
            p, names = self.extra_blocks(p, c, names)
        return collections.OrderedDict(zip(names, p))


def torchvision_serve_inference(
    model, x: typing.List[torch.Tensor], resize_config: core.resizing.ResizeConfig
):
    """A convenience function for performing inference using torchvision
    object detectors in a common way with different resize methods."""
    resized, scales, sizes = core.resizing.resize(x, resize_config=resize_config)
    # Go from [sy, sx] to [sx, sy, sx, sy]
    scales = scales[:, [1, 0]].repeat((1, 2))
    sizes = sizes[:, [1, 0]].repeat((1, 2))
    detections = [
        {k: v.detach().cpu() for k, v in d.items()}
        for d in model.model(resized)["output"]
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
    model: torch.nn.Module,
    anchor_generator,
    device: torch.device,
    height: int,
    width: int,
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
        anchor_generator.sizes  # type: ignore
    ), f"Number of feature maps ({len(feature_maps)}) does not match number of anchor sizes ({len(anchor_generator.sizes)}). This model is misconfigured."  # type: ignore
    return np.concatenate(
        [
            a.cpu()
            for a in anchor_generator(  # type: ignore
                image_list=image_list, feature_maps=list(feature_maps.values())
            )
        ]
    )


def interpret_fpn_kwargs(fpn_kwargs, extra_blocks_kwargs: dict = None):
    """Interpret fpn kwargs to extract extra blocks."""
    interpreted = {
        k: (
            v
            if k != "extra_blocks"
            or isinstance(v, torchvision.ops.feature_pyramid_network.ExtraFPNBlock)
            else EXTRA_BLOCKS_MAP[typing.cast(str, v)]  # type: ignore
        )
        for k, v in fpn_kwargs.items()
    }
    if "extra_blocks" in interpreted and extra_blocks_kwargs is not None:
        interpreted["extra_blocks"] = interpreted["extra_blocks"](**extra_blocks_kwargs)
    return interpreted


def initialize_basic(
    default_map: typing.Dict[str, dict],
    backbone: str,
    fpn_kwargs: dict = None,
    anchor_kwargs: dict = None,
    detector_kwargs: dict = None,
    resize_config: core.resizing.ResizeConfig = None,
    pretrained_backbone: bool = True,
):
    """Initialize basic FPN-based model."""
    fpn_kwargs = {
        **(fpn_kwargs or default_map[backbone]["default_fpn_kwargs"]),
        "pretrained": pretrained_backbone,
    }
    anchor_kwargs = anchor_kwargs or default_map[backbone]["default_anchor_kwargs"]
    detector_kwargs = (
        detector_kwargs or default_map[backbone]["default_detector_kwargs"]
    )
    fpn = default_map[backbone]["fpn_func"](
        **interpret_fpn_kwargs(
            fpn_kwargs,
            extra_blocks_kwargs=default_map[backbone].get("fpn_extra_blocks_kwargs"),
        )
    )
    resize_config = resize_config or {
        "method": "pad_to_multiple",
        "base": 128,
        "max": None,
        "cval": 0,
    }
    # In mira, backbone has meaning because we use it to skip
    # training these weights. But the FPN includes feature extraction
    # layers that we likely we want to change, so we distinguish
    # between the FPN and the backbone.
    backbone = fpn.body
    anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
        **anchor_kwargs
    )
    return (
        resize_config,
        anchor_kwargs,
        fpn_kwargs,
        detector_kwargs,
        backbone,
        fpn,
        anchor_generator,
    )
