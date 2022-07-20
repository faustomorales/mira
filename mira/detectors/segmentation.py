import torch
import cv2
import numpy as np
import pkg_resources
from ..thirdparty.smp import segmentation_models_pytorch as smp
from . import detector as mdd
from .. import core as mc


class SMPWrapper(torch.nn.Module):
    """A wrapper for smp models to make them
    compatible with mira detectors."""

    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        self.backbone = self.model.encoder

    def forward(self, x, targets=None):
        """Perform a forward pass, returning loss
        if in training mode, otherwise returning the
        sigmoid outputs."""
        y = self.model(x)
        if torch.jit.is_tracing():
            return y.sigmoid()
        return {
            "loss": self.loss(y_pred=y.contiguous(), y_true=targets)
            if self.training
            else None,
            "output": [{"map": yi} for yi in y.sigmoid()],
        }


# pylint: disable=too-many-instance-attributes
class SMP(mdd.Detector):
    """A detector that uses segmentation-models-pytorch
    under the hood to build quasi-object-detection models.

    Args:
        categories: A configuration for annotations.
        arch: A segmentation_models_pytorch model class to use.
            Defaults to smp.UnetPlusPlus.
        device: The device to load the model onto.
        encoder_name: The name of the encoder to use with the
            smp model.
        loss: The loss function use. Defaults to
            smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE)
    """

    def __init__(
        self,
        categories: mc.Categories,
        pretrained_backbone: bool = True,
        arch="UnetPlusPlus",
        device="cpu",
        max_detections: int = None,
        detector_kwargs=None,
        backbone_kwargs=None,
        preprocessing_kwargs=None,
        resize_config: mc.resizing.ResizeConfig = None,
        base_threshold: float = 0.5,
    ):
        self.arch = arch
        self.backbone_kwargs = backbone_kwargs or {"encoder_name": "efficientnet-b0"}
        self.detector_kwargs = detector_kwargs or {
            "loss": smp.losses.DiceLoss(
                smp.losses.MULTILABEL_MODE,
            )
        }
        self.max_detections = max_detections
        self.preprocessing_kwargs = preprocessing_kwargs or {
            "encoder_name": self.backbone_kwargs["encoder_name"],
            "pretrained": "imagenet",
        }
        self.resize_config = resize_config or {"method": "pad_to_multiple", "base": 64}
        self.categories = mc.Categories.from_categories(categories)
        self.model = SMPWrapper(
            model=getattr(smp, self.arch)(
                **self.backbone_kwargs,
                classes=len(categories),
                encoder_weights="imagenet" if pretrained_backbone else None,
            ),
            **self.detector_kwargs,
        )
        self.set_device(device)
        self.backbone = self.model.backbone
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            **self.preprocessing_kwargs
        )
        self.base_threshold = base_threshold

    def compute_inputs(self, images):
        return torch.tensor(
            self.preprocessing_fn(np.float32(images)),
            dtype=torch.float32,
            device=self.device,
        ).permute(0, 3, 1, 2)

    def compute_targets(self, annotation_groups, width, height):
        segmaps = np.zeros(
            (len(annotation_groups), len(self.categories), height, width),
            dtype="float32",
        )
        for annotations, segmap in zip(annotation_groups, segmaps):
            for annotation in annotations:
                index = self.categories.index(annotation.category)
                annotation.draw(segmap[index], color=1, opaque=True)
        return torch.tensor(segmaps, device=self.device)

    def invert_targets(self, y, threshold=0.5, **kwargs):
        return [
            mc.utils.flatten(
                [
                    [
                        ann
                        for ann in [
                            mc.Annotation(
                                points=contour[:, 0],
                                category=category,
                                score=catmap[
                                    contour[:, 0, 1]
                                    .min(axis=0) : contour[:, 0, 1]
                                    .max(axis=0)
                                    + 1,
                                    contour[:, 0, 0].min() : contour[:, 0, 0].max() + 1,
                                ].max()
                                if np.product(
                                    contour[:, 0].max(axis=0)
                                    - contour[:, 0].min(axis=0)
                                )
                                > 0
                                else self.base_threshold,
                            )
                            for contour in sorted(
                                cv2.findContours(
                                    (
                                        catmap > min(self.base_threshold, threshold)
                                    ).astype("uint8"),
                                    mode=cv2.RETR_LIST,
                                    method=cv2.CHAIN_APPROX_SIMPLE,
                                )[0],
                                key=cv2.contourArea,
                                reverse=True,
                            )[: self.max_detections]
                        ]
                        if ann.score > threshold
                    ]
                    for catmap, category in zip(
                        segmap["map"].detach().cpu().numpy(), self.categories
                    )
                ]
            )
            for segmap in y["output"]
        ]

    def serve_module_string(self):
        return (
            pkg_resources.resource_string("mira", "detectors/assets/serve/smp.py")
            .decode("utf-8")
            .replace("NUM_CLASSES", str(len(self.categories)))
            .replace("BACKBONE_KWARGS", str(self.backbone_kwargs))
            .replace("RESIZE_CONFIG", str(self.resize_config))
            .replace("DETECTOR_KWARGS", str({**self.detector_kwargs, "loss": None}))
            .replace("PREPROCESSING_KWARGS", str(self.preprocessing_kwargs))
            .replace("ARCH", f"'{self.arch}'")
            .replace("MAX_DETECTIONS", str(self.max_detections))
            .replace("BASE_THRESHOLD", str(self.base_threshold))
        )

    def compute_anchor_boxes(self, width: int, height: int) -> np.ndarray:
        raise ValueError("This detector does not use anchor boxes.")
