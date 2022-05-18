# pylint: disable=too-many-instance-attributes
import typing
import logging
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
from . import common as mdc

LOGGER = logging.getLogger(__name__)


class ModifiedRetinaNet(torchvision.models.detection.retinanet.RetinaNet):
    """Modified version of RetinaNet that always computes inferences."""

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: typing.List[typing.Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        LOGGER.debug("Transformed images to shapes %s", images.tensors.shape)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: typing.List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = collections.OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: typing.List[typing.Dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors)
        force_compute_detections = True
        if not self.training or force_compute_detections:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: typing.Dict[str, typing.List[torch.Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(
                    head_outputs[k].split(num_anchors_per_level, dim=1)
                )
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(
                split_head_outputs, split_anchors, images.image_sizes
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )

        return {
            "loss": sum(loss for loss in losses.values()) if losses else None,
            "output": detections,
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
            p, names = self.extra_blocks(c=c, p=p, names=names)
        return collections.OrderedDict(zip(names, p))


EXTRA_BLOCKS_MAP = {
    "lastlevelp6p7": lambda: torchvision.ops.feature_pyramid_network.LastLevelP6P7,
    "lastlevelp6p7_256": functools.partial(
        torchvision.ops.feature_pyramid_network.LastLevelP6P7,
        in_channels=256,
        out_channels=256,
    ),
    "noop": mdc.LastLevelNoop,
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
        backbone_kwargs=None,
        detector_kwargs=None,
        anchor_kwargs=None,
        resize_config: mdc.ResizeConfig = None,
    ):
        super().__init__()
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
        self.resize_config = resize_config or {"method": "pad_to_multiple", "base": 128}
        # In mira, backbone has meaning because we use it to skip
        # training these weights. But the FPN includes feature extraction
        # layers that we likely we want to change, so we distinguish
        # between the FPN and the backbone.
        self.backbone = self.fpn.body
        self.model = ModifiedRetinaNet(
            backbone=self.fpn,
            num_classes=len(annotation_config) + 1,
            anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **self.anchor_kwargs
            ),
            **self.detector_kwargs,
        )
        self.model.transform = mdc.convert_rcnn_transform(self.model.transform)
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
        self.backbone_name = backbone

    def serve_module_string(self):
        return (
            pkg_resources.resource_string("mira", "detectors/assets/serve/retinanet.py")
            .decode("utf-8")
            .replace("NUM_CLASSES", str(len(self.annotation_config) + 1))
            .replace("BACKBONE_NAME", f"'{self.backbone_name}'")
            .replace("RESIZE_CONFIG", str(self.resize_config))
            .replace("DETECTOR_KWARGS", str(self.detector_kwargs))
            .replace("ANCHOR_KWARGS", str(self.anchor_kwargs))
            .replace(
                "BACKBONE_KWARGS", str({**self.backbone_kwargs, "pretrained": False})
            )
        )

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
            for labels in y["output"]
        ]

    def compute_targets(self, annotation_groups, width, height):
        return [
            {
                "boxes": torch.tensor(b[:, :4], dtype=torch.float32).to(self.device),
                "labels": torch.tensor(b[:, -1] + 1, dtype=torch.int64).to(self.device),
            }
            for b in [
                self.annotation_config.bboxes_from_group(g) for g in annotation_groups
            ]
        ]

    def compute_anchor_boxes(self, width, height):
        return mdc.get_torchvision_anchor_boxes(
            model=self.model, device=self.device, height=height, width=width
        )
