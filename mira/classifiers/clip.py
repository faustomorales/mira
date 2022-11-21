import torch
import torchvision
import numpy as np

import mira.core
from mira.thirdparty.clip import clip
from .classifier import Classifier

DEFAULT_NORMALIZATION = {
    "mean": (0.48145466, 0.4578275, 0.40821073),
    "std": (0.26862954, 0.26130258, 0.27577711),
}


class ModifiedCLIP(torch.nn.Module):
    """A  CLIP wrapper to support simpler calls."""

    def __init__(self, model_name, device):
        super().__init__()
        self.model = clip.load(model_name, device=device)[0]
        self.logit_scale = self.model.logit_scale
        self.exemplar_mapping = None
        self.exemplar_vectors = None

    def forward(self, x):
        """Model forward pass for images."""
        return self.model.encode_image(x)

    def encode_text(self, *args, **kwargs):
        """Encode text classes -- passed to clip.encode_text"""
        return self.model.encode_text(*args, **kwargs)


class CLIP(Classifier):
    """A wrapper for OpenAI's CLIP classifier."""

    def __init__(self, categories, model_name="RN50", device="cpu"):
        self.model = ModifiedCLIP(model_name, device=device)
        self.resize_config = {
            "method": "fit",
            "width": self.model.model.visual.input_resolution,
            "height": self.model.model.visual.input_resolution,
            "cval": 0,
        }
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(**DEFAULT_NORMALIZATION),
            ]
        )
        self.set_device(device)
        self.model_name = model_name
        self.categories = mira.core.Categories.from_categories(categories)
        self.text_vectors = self.model.encode_text(
            clip.tokenize([c.name for c in self.categories]).to(self.device)
        )

    def compute_inputs(self, images: np.ndarray):
        return self.preprocess(super().compute_inputs(images))

    def invert_targets(self, y, threshold=0.0):
        logits = (
            (
                self.model.logit_scale.exp()  # type: ignore
                * (y / y.norm(dim=1, keepdim=True))
                @ (self.text_vectors / self.text_vectors.norm(dim=1, keepdim=True)).t()
            )
            .detach()
            .cpu()
        )
        return mira.core.torchtools.logits2labels(
            logits=logits, categories=self.categories, threshold=threshold
        )

    def compute_targets(self, targets, width, height):
        raise NotImplementedError("CLIP traning is not supported.")
