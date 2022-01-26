import types
import typing

import torch
import numpy as np
import typing_extensions as tx


from . import detector
from .. import datasets as mds
from .. import core as mc
from ..thirdparty.detr import models as dm
from ..thirdparty.detr.util import box_ops as dub


class DETRWrapper(torch.nn.Module):
    """A wrapper for DETR training and inference."""

    def __init__(self, model, criterion, postprocessors):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

    def forward(self, samples, targets=None):
        """Forward pass with return specified by whether we are in training."""
        outputs = self.model(samples)
        if self.training:
            assert targets is not None
            targets_transformed = [
                {
                    # Need cx, cy, w, h, normalized by dimensions.
                    "boxes": dub.box_xyxy_to_cxcywh(t["boxes"])
                    / torch.tensor(
                        samples.shape[-2:][::-1], device=t["boxes"].device
                    ).repeat(2),
                    "labels": t["labels"],
                }
                for t in targets
            ]
            loss_dict = self.criterion(outputs, targets_transformed)
            weight_dict = self.criterion.weight_dict
            return {
                "loss": sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
            }
        assert targets is None
        return self.postprocessors["bbox"](
            outputs,
            torch.tensor(
                np.array([s.shape[-2:] for s in samples]), device=samples.device
            ),
        )


DETRBuildArgs = tx.TypedDict(
    "DETRBuildArgs",
    {
        "dataset_file": typing.Optional[str],
        "device": str,
        "num_queries": int,
        "num_classes": int,
        "frozen_weights": typing.Optional[str],
        "masks": bool,
        "bbox_loss_coef": float,
        "giou_loss_coef": float,
        "mask_loss_coef": typing.Optional[float],
        "dice_loss_coef": typing.Optional[float],
        "aux_loss": bool,
        "dec_layers": int,
        "enc_layers": int,
        "eos_coef": float,
        "hidden_dim": int,
        "position_embedding": tx.Literal["sine", "learned"],
        "lr_backbone": float,
        "backbone": str,
        "dilation": bool,
        "pretrained_backbone": bool,
        "dropout": float,
        "nheads": int,
        "dim_feedforward": int,
        "pre_norm": bool,
        "set_cost_class": int,
        "set_cost_bbox": int,
        "set_cost_giou": int,
        "return_layer": str,
    },
)

DEFAULT_DETR_ARGS = DETRBuildArgs(
    dataset_file=None,
    device="cpu",
    num_queries=100,
    frozen_weights=None,
    masks=False,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    mask_loss_coef=None,
    dice_loss_coef=None,
    aux_loss=False,
    dec_layers=6,
    enc_layers=6,
    eos_coef=0.1,
    hidden_dim=256,
    position_embedding="sine",
    lr_backbone=1e-5,
    backbone="resnet50",
    dilation=False,
    pretrained_backbone=False,
    dropout=0.1,
    nheads=8,
    dim_feedforward=2048,
    pre_norm=False,
    set_cost_class=1,
    set_cost_bbox=5,
    set_cost_giou=2,
    num_classes=100,
    return_layer="layer4",
)


class DETR(detector.Detector):
    """Facebook's DETR model."""

    def __init__(
        self,
        annotation_config=mds.COCOAnnotationConfiguration90,
        build_kwargs: DETRBuildArgs = None,
        pretrained_backbone: bool = True,
        pretrained_top=False,
        device="cpu",
        resize_method: detector.ResizeMethod = "fit",
    ):
        if pretrained_top:
            pretrained_backbone = False
        self.build_kwargs = {
            **DEFAULT_DETR_ARGS,
            **(build_kwargs or {}),
            "num_classes": len(annotation_config) + 1,
            "pretrained_backbone": pretrained_backbone,
            "device": device,
        }
        self.resize_method = resize_method
        self.model = DETRWrapper(*dm.build(types.SimpleNamespace(**self.build_kwargs)))
        self.backbone = self.model.model.backbone
        self.annotation_config = annotation_config
        self.set_input_shape(width=512, height=512)
        self.set_device(torch.device(device))
        if pretrained_top:
            assert (
                annotation_config == mds.COCOAnnotationConfiguration90
            ), "Only COCO is supported for pretrained_top=True"
            self.model.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
                    map_location=self.device,
                    check_hash=True,
                )["model"]
            )

    @property
    def training_model(self):
        """The training model."""
        return self.model

    @property
    def anchor_boxes(self):
        return None

    def compute_inputs(self, images):
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        images = (np.float32(images) - mean) / std
        return (
            torch.tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

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
    def input_shape(self):
        return self._input_shape

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

    @property
    def serve_module_index(self):
        raise NotImplementedError

    def serve_module_string(self, enable_flexible_size=False):
        raise NotImplementedError

    def set_input_shape(self, width, height):
        self._input_shape = (height, width, 3)
