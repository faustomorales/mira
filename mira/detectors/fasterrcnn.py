import torch
import torchvision
import numpy as np
import typing_extensions as tx

from .. import datasets as mds
from .. import core as mc
from .detector import Detector


BACKBONE_TO_CONSTRUCTOR = {
    "resnet50": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "mobilenet_large": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "mobilenet_large_320": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
}


class FasterRCNN(Detector):
    """A wrapper around the FasterRCNN models in torchvision."""

    def __init__(
        self,
        annotation_config=mds.COCOAnnotationConfiguration90,
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        backbone: tx.Literal[
            "resnet50", "mobilenet_large", "mobilenet_large_320"
        ] = "resnet50",
    ):
        self.annotation_config = annotation_config
        self.model = BACKBONE_TO_CONSTRUCTOR[backbone](
            pretrained=pretrained_top,
            progress=True,
            num_classes=len(annotation_config) + 1,
            pretrained_backbone=pretrained_backbone,
        )
        self.set_input_shape(width=512, height=512)

    def compute_inputs(self, images):
        images = np.float32(images) / 255.0
        return torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)

    def invert_targets(self, y, threshold=0.5, **kwargs):
        return [
            [
                mc.Annotation(
                    category=self.annotation_config[int(labelIdx) - 1],
                    selection=mc.Selection(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=score,
                )
                for (x1, y1, x2, y2), labelIdx, score in zip(
                    labels["boxes"], labels["labels"], labels["scores"]
                )
                if score > threshold
            ]
            for labels in y
        ]

    def set_input_shape(self, width, height):
        self._input_shape = (height, width, 3)

    @property
    def input_shape(self):
        return self._input_shape

    def compute_targets(self, annotation_groups):
        return [
            {
                "boxes": torch.tensor(b[:, :4], dtype=torch.float32),
                "labels": torch.tensor(b[:, -1] + 1),
            }
            for b in [
                self.annotation_config.bboxes_from_group(g) for g in annotation_groups
            ]
        ]
