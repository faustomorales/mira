# pylint: disable=too-many-instance-attributes
import typing

import torch
import torchvision
import typing_extensions as tx
import pkg_resources

from .. import datasets as mds
from .. import core as mc
from . import detector
from . import common as mdc


BACKBONE_TO_PARAMS = {
    "resnet50": {
        "weights_url": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "fpn_func": torchvision.models.detection.backbone_utils.resnet_fpn_backbone,
        "fpn_extra_blocks_kwargs": {},
        "default_fpn_kwargs": {
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
    "timm": {
        "fpn_func": mdc.BackboneWithTIMM,
        "default_fpn_kwargs": {
            "model_name": "mnasnet_small",
            "out_indices": (2, 3, 4),
        },
        "default_anchor_kwargs": {
            "sizes": tuple(
                (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                for x in [32, 64, 128]
            ),
            "aspect_ratios": ((1.0,),) * 3,
        },
        "default_detector_kwargs": {},
    },
    "mobilenet_large": {
        "weights_url": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        "fpn_extra_blocks_kwargs": {},
        "fpn_func": torchvision.models.detection.backbone_utils.mobilenet_backbone,
        "default_fpn_kwargs": {
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
        "fpn_extra_blocks_kwargs": {},
        "fpn_func": torchvision.models.detection.backbone_utils.mobilenet_backbone,
        "default_fpn_kwargs": {
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


class ModifiedFasterRCNN(torchvision.models.detection.faster_rcnn.FasterRCNN):
    """Modified version of Faster RCNN that always computes inferences."""

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        original_image_sizes: typing.List[typing.Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

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

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = typing.OrderedDict([("0", features)])
        else:
            # Forces features to comply with self.roi_heads.box_roi_pool.featmap_names
            features = typing.OrderedDict(
                [(str(idx), f) for idx, f in enumerate(features.values())]
            )
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.training:
            self.roi_heads.training = False
            detections, _ = self.roi_heads(features, proposals, images.image_sizes)
            self.roi_heads.training = True
            _, detector_losses = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )
        else:
            detections, detector_losses = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return {
            "loss": sum(loss for loss in losses.values()) if losses else None,
            "output": detections,
        }


class FasterRCNN(detector.Detector):
    """A wrapper around the FasterRCNN models in torchvision."""

    def __init__(
        self,
        categories=mds.COCOCategories90,
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        backbone: tx.Literal[
            "resnet50", "mobilenet_large", "mobilenet_large_320"
        ] = "resnet50",
        device="cpu",
        fpn_kwargs=None,
        detector_kwargs=None,
        anchor_kwargs=None,
        resize_config: mc.resizing.ResizeConfig = None,
    ):
        self.categories = mc.Categories.from_categories(categories)
        self.backbone_name = backbone
        if pretrained_top:
            pretrained_backbone = False
        (
            self.resize_config,
            self.anchor_kwargs,
            self.fpn_kwargs,
            self.detector_kwargs,
            self.backbone,
            fpn,
            anchor_generator,
        ) = mdc.initialize_basic(
            BACKBONE_TO_PARAMS,
            backbone=backbone,
            fpn_kwargs=fpn_kwargs,
            anchor_kwargs=anchor_kwargs,
            detector_kwargs=detector_kwargs,
            resize_config=resize_config,
            pretrained_backbone=pretrained_backbone,
        )
        self.model = ModifiedFasterRCNN(
            fpn,
            len(categories) + 1,
            rpn_anchor_generator=anchor_generator,
            **self.detector_kwargs,
        )
        self.model.transform = mdc.convert_rcnn_transform(self.model.transform)
        if pretrained_top:
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    BACKBONE_TO_PARAMS[backbone]["weights_url"],
                    progress=True,
                )
            )
            torchvision.models.detection.faster_rcnn.overwrite_eps(self.model, 0.0)
        self.set_device(device)

    def serve_module_string(self):
        return (
            pkg_resources.resource_string(
                "mira", "detectors/assets/serve/fasterrcnn.py"
            )
            .decode("utf-8")
            .replace("NUM_CLASSES", str(len(self.categories) + 1))
            .replace("BACKBONE_NAME", f"'{self.backbone_name}'")
            .replace("RESIZE_CONFIG", str(self.resize_config))
            .replace("DETECTOR_KWARGS", str(self.detector_kwargs))
            .replace("ANCHOR_KWARGS", str(self.anchor_kwargs))
            .replace("FPN_KWARGS", str({**self.fpn_kwargs, "pretrained": False}))
        )

    def invert_targets(self, y, threshold=0.5, **kwargs):
        return [
            [
                mc.Annotation(
                    category=self.categories[int(labelIdx) - 1],
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
            for b in [self.categories.bboxes_from_group(g) for g in annotation_groups]
        ]

    def compute_anchor_boxes(self, width, height):
        return mdc.get_torchvision_anchor_boxes(
            model=self.model,
            anchor_generator=typing.cast(
                torch.nn.Module, self.model.rpn
            ).anchor_generator,
            device=self.device,
            height=height,
            width=width,
        )
