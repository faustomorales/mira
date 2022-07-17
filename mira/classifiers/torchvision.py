import typing

import numpy as np
import torch
import torchvision

from .. import core as mc
from .classifier import Classifier
from ..datasets.preloaded import ImageNet1KAnnotationConfiguration


class TVW(torch.nn.Module):
    """A module wrapper for torchvision models."""

    def __init__(
        self,
        num_classes,
        initializer: typing.Any,
        weights: typing.Any,
        pretrained_top: bool,
        dropout=0.5,
    ):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = initializer(
            weights=weights,
            progress=True,
        )
        if not pretrained_top:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=True),
                torch.nn.Linear(
                    list(self.model.classifier.children())[-1].in_features,
                    num_classes,
                ),
            )
        else:
            assert (
                list(self.model.classifier.children())[-1].out_features == num_classes
            )
        if hasattr(self.model, "features"):
            self.backbone = self.model.features
        else:
            self.backbone = self.model.layers
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        """Perform forward pass on inputs."""
        y = self.model(self.transform(x))
        return {
            "output": y,
            "loss": self.loss(y, targets) if targets is not None else None,
        }


class TorchVisionClassifier(Classifier):
    """A classifier built on top of the torchvision classifiers."""

    def __init__(
        self,
        annotation_config=ImageNet1KAnnotationConfiguration,
        model_name="efficientnet_b0",
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        pretrained_top=False,
        resize_config: mc.torchtools.ResizeConfig = None,
        device="cpu",
    ):
        self.resize_config = resize_config or {
            "method": "fit",
            "width": 224,
            "height": 224,
        }
        self.annotation_config = annotation_config
        self.model = TVW(
            num_classes=len(self.annotation_config),
            initializer=getattr(torchvision.models, model_name),
            weights=weights,
            pretrained_top=pretrained_top,
        )
        self.backbone = self.model.backbone
        self.set_device(device)

    def invert_targets(self, y):
        logits = y["output"].detach().cpu()
        scores = logits.softmax(dim=-1).numpy()
        return [
            {
                "label": mc.Label(
                    category=self.annotation_config[classIdx],
                    score=score,
                ),
                "logit": logit,
                "raw": {
                    category.name: {"score": score, "logit": logit}
                    for category, score, logit in zip(
                        self.annotation_config,
                        catscores.tolist(),
                        catlogits.tolist(),
                    )
                },
            }
            for score, classIdx, logit, catscores, catlogits in zip(
                scores.max(axis=1).tolist(),
                scores.argmax(axis=1).tolist(),
                logits.numpy().max(axis=1).tolist(),
                scores,
                logits.numpy(),
            )
        ]

    def compute_targets(
        self,
        label_groups,
    ):
        y = np.zeros((len(label_groups), len(self.annotation_config)), dtype="float32")
        for bi, labels in enumerate(label_groups):
            for label in labels:
                y[bi, self.annotation_config.index(label.category)] = 1
        return torch.Tensor(y).to(self.device)
