import cv2
import timm
import torch
import numpy as np

from ... import core as mc
from .. import detector as md

# pylint: disable=missing-function-docstring
class PixelwiseClassifier(torch.nn.Module):
    """Basic segmentation model."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=n_features, out_channels=n_classes, kernel_size=1, bias=True
        )
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        return y


def flatten(t):
    """Standard utility function for flattening a nested list."""
    return [item for sublist in t for item in sublist]


class Segmenter(torch.nn.Module):
    """A classifier using segmentation on bounding boxes."""

    def __init__(
        self,
        n_classes=5,
        downsample=4,
        model_name="efficientnet_lite0",
        pretrained=True,
        positive_agg=torch.max,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=True
        )
        self.downsample = downsample
        self.classifier = PixelwiseClassifier(
            n_features=sum(
                [f.shape[1] for f in self.backbone(torch.rand(1, 3, 128, 128))]
            ),
            n_classes=n_classes,
        )
        self.positive_agg = positive_agg
        self.eval()

    def forward(self, x, target=None):
        y = self.backbone(x)
        y = [
            torch.nn.functional.interpolate(
                f, size=(x.shape[2] // self.downsample, x.shape[3] // self.downsample)
            )
            for f in y
        ]
        y = torch.cat(y, dim=1)
        y = self.classifier(y)
        if self.training:
            binary_inputs_list = []
            if any(ti["negative_pixels"] for ti in target):
                binary_inputs_list.append(
                    torch.cat(
                        flatten(
                            [
                                [
                                    torch.unsqueeze(
                                        torch.masked_select(yi[c], mask).max(), 0
                                    )
                                    for mask, c in ti["negative_pixels"]
                                ]
                                for yi, ti in zip(y, target)
                            ]
                        )
                    )
                )
            if any(ti["positive_bboxes"] for ti in target):
                binary_inputs_list.append(
                    torch.cat(
                        flatten(
                            [
                                [
                                    torch.unsqueeze(
                                        self.positive_agg(yi[c, y1:y2, x1:x2]), 0  # type: ignore
                                    )
                                    for x1, y1, x2, y2, c in ti["positive_bboxes"]
                                ]
                                for yi, ti in zip(y, target)
                            ]
                        )
                    )
                )
            binary_inputs = torch.cat(binary_inputs_list)
            binary_target = torch.cat(
                [
                    torch.cat(
                        [
                            torch.zeros(len(ti["negative_pixels"]), device=x.device)
                            for ti in target
                        ]
                    ),
                    torch.cat(
                        [
                            torch.ones(len(ti["positive_bboxes"]), device=x.device)
                            for ti in target
                        ]
                    ),
                ]
            )
            return {
                "loss": torch.nn.functional.binary_cross_entropy(
                    input=binary_inputs, target=binary_target, reduction="mean"
                )
            }
        return y


def compute_boxes(scores, threshold):
    binary = (scores > threshold).astype("uint8")
    contours = cv2.findContours(
        binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )[0]
    x1y1x2y2 = np.array(
        [[c[:, 0].min(axis=0), c[:, 0].max(axis=0) + 1] for c in contours]
    ).reshape(-1, 4)
    return [
        (x1, y1, x2, y2, scores[y1:y2, x1:x2].mean()) for x1, y1, x2, y2 in x1y1x2y2
    ]


class AggregatedSegmentation(md.Detector):
    """A custom detector using segmentation to detect bounding boxes."""

    def __init__(
        self,
        annotation_config,
        device="cpu",
        resize_method: md.ResizeMethod = "fit",
        downsample: int = 4,
        model_name="efficientnet_lite0",
        pretrained_backbone: bool = True,
        positive_agg=torch.max,
    ):
        super().__init__(device=device, resize_method=resize_method)
        self.annotation_config = annotation_config
        self.model = Segmenter(
            n_classes=len(annotation_config),
            downsample=downsample,
            model_name=model_name,
            pretrained=pretrained_backbone,
            positive_agg=positive_agg,
        ).to(self.device)
        self.set_input_shape(width=256, height=256)

    @property
    def training_model(self):
        """Training model for this detector."""
        return self.model

    def compute_inputs(self, images):
        images = np.float32(images) / 255.0
        images -= self.model.backbone.default_cfg["mean"]  # type: ignore
        images /= self.model.backbone.default_cfg["std"]  # type: ignore
        return (
            torch.tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

    def invert_targets(self, y, threshold=0.5, **kwargs):
        annotations = []
        ds = self.model.downsample
        for yi in y.detach().cpu().numpy():
            current_annotations = []
            for labelIdx, category in enumerate(self.annotation_config):
                current_annotations.extend(
                    [
                        mc.Annotation(
                            category=category,
                            x1=x1 * ds,
                            y1=y1 * ds,
                            x2=x2 * ds,
                            y2=y2 * ds,
                            score=score,
                        )
                        for x1, y1, x2, y2, score in compute_boxes(
                            yi[labelIdx], threshold=threshold
                        )
                    ]
                )
            annotations.append(current_annotations)
        return annotations

    def set_input_shape(self, width, height):
        self._input_shape = (height, width, 3)

    @property
    def serve_module_string(self):
        raise NotImplementedError

    @property
    def serve_module_index(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._input_shape

    def compute_targets(self, annotation_groups):
        height, width = self.input_shape[:2]
        ds = self.model.downsample
        height_ds, width_ds = height // ds, width // ds
        targets = []
        for g in annotation_groups:
            positive_bboxes = [
                (x1 // ds, y1 // ds, x2 // ds, y2 // ds, cIdx)
                for x1, y1, x2, y2, cIdx in self.annotation_config.bboxes_from_group(g)
            ]
            negative_pixels = np.ones(
                (len(self.annotation_config), height_ds, width_ds), dtype="bool"
            )
            for x1, y1, x2, y2, cIdx in positive_bboxes:
                negative_pixels[cIdx, y1:y2, x1:x2] = False
            targets.append(
                {
                    "positive_bboxes": positive_bboxes,
                    "negative_pixels": [
                        (torch.tensor(p).to(self.device), cIdx)
                        for cIdx, p in enumerate(negative_pixels)
                        if p.any()
                    ],
                }
            )
        return targets
