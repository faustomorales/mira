import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from . import detector as mdd
from . import common as mdc
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
        annotation_config: A configuration for annotations.
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
        annotation_config: mc.AnnotationConfiguration,
        pretrained_backbone: bool = True,
        arch="UnetPlusPlus",
        device="cpu",
        max_detections: int = None,
        detector_kwargs=None,
        backbone_kwargs=None,
        preprocessing_fn=None,
        resize_config: mdc.ResizeConfig = None,
    ):
        self.backbone_kwargs = {
            "encoder_name": "efficientnet-b0",
            **(backbone_kwargs or {}),
        }
        self.detector_kwargs = {
            "loss": smp.losses.DiceLoss(
                smp.losses.MULTILABEL_MODE,
            ),
            **(detector_kwargs or {}),
        }
        self.model = SMPWrapper(
            model=getattr(smp, arch)(
                **self.backbone_kwargs,
                classes=len(annotation_config),
                encoder_weights="imagenet" if pretrained_backbone else None,
            ),
            **self.detector_kwargs,
        )
        self.backbone = self.model.backbone
        self.set_device(device)
        self.annotation_config = annotation_config
        self.max_detections = max_detections
        self.preprocessing_fn = preprocessing_fn or smp.encoders.get_preprocessing_fn(
            self.backbone_kwargs["encoder_name"], pretrained="imagenet"
        )
        self.resize_config = resize_config or {"method": "pad_to_multiple", "base": 64}

    def compute_inputs(self, images):
        return torch.tensor(
            self.preprocessing_fn(np.float32(images)),
            dtype=torch.float32,
            device=self.device,
        ).permute(0, 3, 1, 2)

    def compute_targets(self, annotation_groups, width, height):
        segmaps = np.zeros(
            (len(annotation_groups), len(self.annotation_config), height, width),
            dtype="float32",
        )
        for annotations, segmap in zip(annotation_groups, segmaps):
            for annotation in annotations:
                index = self.annotation_config.index(annotation.category)
                annotation.draw(segmap[index], color=1, opaque=True)
        return torch.tensor(segmaps, device=self.device)

    def invert_targets(self, y, threshold=0.5, **kwargs):
        return [
            mc.utils.flatten(
                [
                    [
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
                                contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0)
                            )
                            > 0
                            else threshold,
                        )
                        for contour in sorted(
                            cv2.findContours(
                                (catmap > threshold).astype("uint8"),
                                mode=cv2.RETR_LIST,
                                method=cv2.CHAIN_APPROX_SIMPLE,
                            )[0],
                            key=cv2.contourArea,
                            reverse=True,
                        )[: self.max_detections]
                    ]
                    for catmap, category in zip(
                        segmap["map"].detach().cpu().numpy(), self.annotation_config
                    )
                ]
            )
            for segmap in y["output"]
        ]

    def serve_module_string(self):
        raise NotImplementedError
