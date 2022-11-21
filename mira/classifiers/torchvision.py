import typing

import torch
import torchvision
import numpy as np

from .. import core
from .classifier import Classifier
from ..datasets.preloaded import ImageNet1KCategories


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
            "output": [{"logits": yi} for yi in y],
            "loss": self.loss(y, targets) if targets is not None else None,
        }


class TorchVisionClassifier(Classifier):
    """A classifier built on top of the torchvision classifiers."""

    def __init__(
        self,
        categories=ImageNet1KCategories,
        model_name="efficientnet_b0",
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        pretrained_top=False,
        resize_config: core.resizing.ResizeConfig = None,
        device="cpu",
    ):
        self.resize_config = resize_config or {
            "method": "fit",
            "width": 224,
            "height": 224,
            "cval": 0,
        }
        self.categories = core.Categories.from_categories(categories)
        self.model = TVW(
            num_classes=len(self.categories),
            initializer=getattr(torchvision.models, model_name),
            weights=weights,
            pretrained_top=pretrained_top,
        )
        self.backbone = self.model.backbone
        self.set_device(device)

    def invert_targets(self, y, threshold=0.0):
        logits = torch.stack([o["logits"].detach().cpu() for o in y["output"]], axis=0)  # type: ignore
        return core.torchtools.logits2labels(
            logits=logits, categories=self.categories, threshold=threshold
        )

    def compute_targets(self, targets, width, height):
        return torch.tensor(
            np.stack(
                [
                    core.annotation.labels2onehot(
                        labels=t.labels, categories=self.categories, binary=True
                    )
                    for t in targets
                ],
                axis=0,
            ),
            device=self.device,
        )
