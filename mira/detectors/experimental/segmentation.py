import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from .. import detector as mdd
from ... import core as mc


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
        if self.training:
            return {"loss": self.loss(y_pred=y.contiguous(), y_true=targets)}
        return y.sigmoid()


# pylint: disable=too-many-instance-attributes
class SMP(mdd.Detector):
    """A detector that uses segmentation-models-pytorch
    under the hood to build quasi-object-detection models.

    Args:
        annotation_config: A configuration for annotations.
        arch: A segmentation_models_pytorch model class to use.
            Defaults to smp.UnetPlusPlus.
        device: The device to load the model onto.
        resize_method: The method to use for resizing images.
        encoder_name: The name of the encoder to use with the
            smp model.
        loss: The loss function use. Defaults to
            smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE)
    """

    def __init__(
        self,
        annotation_config: mc.AnnotationConfiguration,
        arch=None,
        device="cpu",
        resize_method="pad_to_multiple",
        encoder_name="efficientnet-b0",
        loss=None,
    ):
        if arch is None:
            arch = smp.UnetPlusPlus
        self.resize_method = resize_method
        self.model = SMPWrapper(
            model=arch(encoder_name=encoder_name, classes=len(annotation_config)),
            loss=loss
            or smp.losses.DiceLoss(
                smp.losses.MULTILABEL_MODE,
            ),
        )
        self.training_model = self.model
        self.backbone = self.model.backbone
        self.set_device(device)
        self.annotation_config = annotation_config
        self.resize_base = 64
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder_name, pretrained="imagenet"
        )
        self.set_input_shape(None, None)

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

    @property
    def input_shape(self):
        return self._input_shape

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
                        for contour in cv2.findContours(
                            (catmap > threshold).astype("uint8"),
                            mode=cv2.RETR_LIST,
                            method=cv2.CHAIN_APPROX_SIMPLE,
                        )[0]
                    ]
                    for catmap, category in zip(
                        segmap.detach().cpu().numpy(), self.annotation_config
                    )
                ]
            )
            for segmap in y
        ]

    @property
    def serve_module_index(self):
        raise NotImplementedError

    def serve_module_string(self, enable_flexible_size=False):
        raise NotImplementedError

    def set_input_shape(self, width, height):
        self._input_shape = (height, width, 3)
