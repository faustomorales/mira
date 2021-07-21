import torch
import torchvision
import numpy as np
import typing_extensions as tx
import pkg_resources

from .. import datasets as mds
from .. import core as mc
from . import detector


BACKBONE_TO_CONSTRUCTOR = {
    "resnet50": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "mobilenet_large": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "mobilenet_large_320": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
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
    ):
        super().__init__(device=device, resize_method=resize_method)
        self.annotation_config = annotation_config
        self.model = BACKBONE_TO_CONSTRUCTOR[backbone](
            pretrained=pretrained_top,
            progress=True,
            num_classes=len(annotation_config) + 1,
            pretrained_backbone=pretrained_backbone,
        ).to(self.device)
        self.set_input_shape(
            width=min(self.model.transform.min_size),  # type: ignore
            height=min(self.model.transform.min_size),  # type: ignore
        )
        self.backbone_name = backbone

    @property
    def training_model(self):
        """Training model for this detector."""
        return self.model

    @property
    def serve_module_string(self):
        return (
            pkg_resources.resource_string(
                "mira", "detectors/assets/serve/fasterrcnn.py"
            )
            .decode("utf-8")
            .replace("NUM_CLASSES", str(len(self.annotation_config) + 1))
            .replace("INPUT_WIDTH", str(self.input_shape[1]))
            .replace("INPUT_HEIGHT", str(self.input_shape[0]))
            .replace("BACKBONE_NAME", f"'{self.backbone_name}'")
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
