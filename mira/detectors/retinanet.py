# pylint: disable=too-many-instance-attributes
import typing
import functools
import collections


import torch
import timm
import torchvision
import numpy as np
import pkg_resources
import typing_extensions as tx

from .. import datasets as mds
from .. import core as mc
from . import detector
from . import common


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
            p, names = self.extra_blocks(c=c, p=p, names=names)
        return collections.OrderedDict(zip(names, p))


EXTRA_BLOCKS_MAP = {
    "lastlevelp6p7": lambda: torchvision.ops.feature_pyramid_network.LastLevelP6P7,
    "lastlevelp6p7_256": functools.partial(
        torchvision.ops.feature_pyramid_network.LastLevelP6P7,
        in_channels=256,
        out_channels=256,
    ),
    "noop": common.LastLevelNoop,
}

DEFAULT_ANCHOR_KWARGS = {
    "sizes": tuple(
        (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
        for x in [32, 64, 128, 256, 512]
    ),
    "aspect_ratios": ((0.5, 1.0, 2.0),) * 5,
}

BACKBONE_TO_PARAMS = {
    "resnet50": {
        "weights_url": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
        "backbone_func": torchvision.models.detection.backbone_utils.resnet_fpn_backbone,
        "default_backbone_kwargs": {
            "trainable_layers": 3,
            "backbone_name": "resnet50",
            "extra_blocks": "lastlevelp6p7_256",
            "returned_layers": [2, 3, 4],
        },
        "default_anchor_kwargs": DEFAULT_ANCHOR_KWARGS,
        "default_detector_kwargs": {},
    },
    "timm": {
        "backbone_func": BackboneWithTIMM,
        "default_backbone_kwargs": {
            "model_name": "efficientnet_b0",
            "out_indices": (0, 1, 2, 3, 4),
        },
        "default_anchor_kwargs": DEFAULT_ANCHOR_KWARGS,
        "default_detector_kwargs": {},
    },
}


class RetinaNet(detector.Detector):
    """A wrapper around the FasterRCNN models in torchvision."""

    def __init__(
        self,
        annotation_config=mds.COCOAnnotationConfiguration90,
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        backbone: tx.Literal["resnet50"] = "resnet50",
        device="cpu",
        resize_method: detector.ResizeMethod = "fit",
        backbone_kwargs=None,
        detector_kwargs=None,
        anchor_kwargs=None,
    ):
        self.resize_method = resize_method
        self.annotation_config = annotation_config
        if pretrained_top:
            pretrained_backbone = False
        self.backbone_kwargs = {
            **BACKBONE_TO_PARAMS[backbone]["default_backbone_kwargs"],
            **(backbone_kwargs or {}),
            "pretrained": pretrained_backbone,
        }
        self.anchor_kwargs = {
            **BACKBONE_TO_PARAMS[backbone]["default_anchor_kwargs"],
            **(anchor_kwargs or {}),
        }
        self.detector_kwargs = {
            **BACKBONE_TO_PARAMS[backbone]["default_detector_kwargs"],
            **(detector_kwargs or {}),
        }
        self.fpn = BACKBONE_TO_PARAMS[backbone]["backbone_func"](
            **{
                k: (
                    v
                    if k != "extra_blocks"
                    or isinstance(
                        v, torchvision.ops.feature_pyramid_network.ExtraFPNBlock
                    )
                    else EXTRA_BLOCKS_MAP[typing.cast(str, v)]()  # type: ignore
                )
                for k, v in self.backbone_kwargs.items()
            }
        )
        # In mira, backbone has meaning because we use it to skip
        # training these weights. But the FPN includes feature extraction
        # layers that we likely we want to change, so we distinguish
        # between the FPN and the backbone.
        self.backbone = self.fpn.body
        self.model = torchvision.models.detection.retinanet.RetinaNet(
            backbone=self.fpn,
            num_classes=len(annotation_config) + 1,
            anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **self.anchor_kwargs
            ),
            **self.detector_kwargs,
        )
        if pretrained_top:
            if "weights_url" not in BACKBONE_TO_PARAMS[backbone]:
                raise ValueError(
                    f"There are no pretrained weights for backbone: {backbone}."
                )
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    BACKBONE_TO_PARAMS[backbone]["weights_url"],
                    progress=True,
                )
            )
            torchvision.models.detection.retinanet.overwrite_eps(self.model, 0.0)
        self.set_device(device)
        self.set_input_shape(
            width=min(self.model.transform.min_size),  # type: ignore
            height=min(self.model.transform.min_size),  # type: ignore
        )
        self.backbone_name = backbone

    @property
    def training_model(self):
        """Training model for this detector."""
        return self.model

    def serve_module_string(self, enable_flexible_size=False):
        return (
            pkg_resources.resource_string("mira", "detectors/assets/serve/retinanet.py")
            .decode("utf-8")
            .replace("NUM_CLASSES", str(len(self.annotation_config) + 1))
            .replace("INPUT_WIDTH", str(self.input_shape[1]))
            .replace("INPUT_HEIGHT", str(self.input_shape[0]))
            .replace("BACKBONE_NAME", f"'{self.backbone_name}'")
            .replace(
                "RESIZE_METHOD",
                "None" if enable_flexible_size else f"'{self.resize_method}'",
            )
            .replace("DETECTOR_KWARGS", str(self.detector_kwargs))
            .replace("ANCHOR_KWARGS", str(self.anchor_kwargs))
            .replace(
                "BACKBONE_KWARGS", str({**self.backbone_kwargs, "pretrained": False})
            )
        )

    @property
    def serve_module_index(self):
        return {
            **{0: "__background__"},
            **{
                str(idx + 1): label.name
                for idx, label in enumerate(self.annotation_config)
            },
        }

    def compute_inputs(self, images):
        images = np.float32(images) / 255.0
        return (
            torch.tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

    def invert_targets(self, y, threshold=0.5, **kwargs):
        return [
            [
                mc.Annotation(
                    category=self.annotation_config[int(labelIdx) - 1],
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                )
                for (x1, y1, x2, y2), labelIdx, score in zip(
                    labels["boxes"].detach().cpu().numpy(),
                    labels["labels"].detach().cpu().numpy(),
                    labels["scores"].detach().cpu().numpy(),
                )
                if score > threshold
            ]
            for labels in y
        ]

    def set_input_shape(self, width, height):
        self._input_shape = (height, width, 3)
        self.model.transform.fixed_size = (height, width)  # type: ignore
        self.model.transform.min_size = (min(width, height),)  # type: ignore
        self.model.transform.max_size = max(height, width)  # type: ignore

    @property
    def input_shape(self):
        return self._input_shape

    def compute_targets(self, annotation_groups):
        return [
            {
                "boxes": torch.tensor(b[:, :4], dtype=torch.float32).to(self.device),
                "labels": torch.tensor(b[:, -1] + 1, dtype=torch.int64).to(self.device),
            }
            for b in [
                self.annotation_config.bboxes_from_group(g) for g in annotation_groups
            ]
        ]

    @property
    def anchor_boxes(self):
        image_list = torchvision.models.detection.image_list.ImageList(
            tensors=torch.tensor(
                np.random.randn(1, *self.input_shape).transpose(0, 3, 1, 2),
                dtype=torch.float32,
            ).to(self.device),
            image_sizes=[self.input_shape[:2]],
        )
        feature_maps = self.model.backbone(image_list.tensors)  # type: ignore
        assert len(feature_maps) == len(
            self.model.anchor_generator.sizes  # type: ignore
        ), f"Number of feature maps ({len(feature_maps)}) does not match number of anchor sizes ({len(self.model.anchor_generator.sizes)}). This model is misconfigured."  # type: ignore
        return np.concatenate(
            [
                a.cpu()
                for a in self.model.anchor_generator(  # type: ignore
                    image_list=image_list, feature_maps=list(feature_maps.values())
                )
            ]
        )
