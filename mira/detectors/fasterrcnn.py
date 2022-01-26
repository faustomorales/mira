# pylint: disable=too-many-instance-attributes
import typing

import torch
import torchvision
import numpy as np
import typing_extensions as tx
import pkg_resources

from .. import datasets as mds
from .. import core as mc
from . import detector
from . import common


EXTRA_BLOCKS_MAP = {
    "lastlevelmaxpool": torchvision.ops.feature_pyramid_network.LastLevelMaxPool,
    "noop": common.LastLevelNoop,
}

BACKBONE_TO_PARAMS = {
    "resnet50": {
        "weights_url": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "backbone_func": torchvision.models.detection.backbone_utils.resnet_fpn_backbone,
        "default_backbone_kwargs": {
            "trainable_layers": 3,
            "backbone_name": "resnet50",
            "extra_blocks": "lastlevelmaxpool",
        },
        "default_anchor_kwargs": {
            "sizes": ((32,), (64,), (128,), (256,), (512,)),
            "aspect_ratios": ((0.5, 1.0, 2.0),) * 5,
        },
        "default_detector_kwargs": {},
    },
    "mobilenet_large": {
        "weights_url": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        "backbone_func": torchvision.models.detection.backbone_utils.mobilenet_backbone,
        "default_backbone_kwargs": {
            "trainable_layers": 3,
            "backbone_name": "mobilenet_v3_large",
            "fpn": True,
            "extra_blocks": "lastlevelmaxpool",
        },
        "default_anchor_kwargs": {
            "sizes": (
                (
                    32,
                    64,
                    128,
                    256,
                    512,
                ),
            )
            * 3,
            "aspect_ratios": ((0.5, 1.0, 2.0),) * 3,
        },
        "default_detector_kwargs": {
            "rpn_score_thresh": 0.05,
        },
    },
    "mobilenet_large_320": {
        "weights_url": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        "backbone_func": torchvision.models.detection.backbone_utils.mobilenet_backbone,
        "default_backbone_kwargs": {
            "trainable_layers": 3,
            "backbone_name": "mobilenet_v3_large",
            "fpn": True,
            "extra_blocks": "lastlevelmaxpool",
        },
        "default_anchor_kwargs": {
            "sizes": (
                (
                    32,
                    64,
                    128,
                    256,
                    512,
                ),
            )
            * 3,
            "aspect_ratios": ((0.5, 1.0, 2.0),) * 3,
        },
        "default_detector_kwargs": {
            "min_size": 320,
            "max_size": 640,
            "rpn_pre_nms_top_n_test": 150,
            "rpn_post_nms_top_n_test": 150,
            "rpn_score_thresh": 0.05,
        },
    },
}


class FasterRCNN(detector.Detector):
    """A wrapper around the FasterRCNN models in torchvision."""

    def __init__(
        self,
        annotation_config=mds.COCOAnnotationConfiguration90,
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        backbone: tx.Literal[
            "resnet50", "mobilenet_large", "mobilenet_large_320"
        ] = "resnet50",
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
        self.backbone = BACKBONE_TO_PARAMS[backbone]["backbone_func"](
            **{
                k: (
                    v
                    if k != "extra_blocks"
                    or isinstance(
                        v, torchvision.ops.feature_pyramid_network.ExtraFPNBlock
                    )
                    else EXTRA_BLOCKS_MAP[typing.cast(str, v)]()
                )
                for k, v in self.backbone_kwargs.items()
            }
        )
        self.model = torchvision.models.detection.faster_rcnn.FasterRCNN(
            self.backbone,
            len(annotation_config) + 1,
            rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **self.anchor_kwargs
            ),
            **self.detector_kwargs,
        )
        if pretrained_top:
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    BACKBONE_TO_PARAMS[backbone]["weights_url"],
                    progress=True,
                )
            )
            torchvision.models.detection.faster_rcnn.overwrite_eps(self.model, 0.0)
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
            pkg_resources.resource_string(
                "mira", "detectors/assets/serve/fasterrcnn.py"
            )
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
        return np.concatenate(
            [
                a.cpu()
                for a in self.model.rpn.anchor_generator(  # type: ignore
                    image_list=image_list, feature_maps=list(feature_maps.values())
                )
            ]
        )
