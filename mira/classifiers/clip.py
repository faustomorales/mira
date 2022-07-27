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

    def invert_targets(self, y):
        logits = (
            (
                self.model.logit_scale.exp()  # type: ignore
                * (y / y.norm(dim=1, keepdim=True))
                @ (self.text_vectors / self.text_vectors.norm(dim=1, keepdim=True)).t()
            )
            .detach()
            .cpu()
        )
        scores = logits.softmax(dim=-1).numpy()
        return [
            [
                mira.core.Label(
                    category=self.categories[classIdx],
                    score=score,
                    metadata={
                        "logit": logit,
                        "raw": {
                            category.name: {"score": score, "logit": logit}
                            for category, score, logit in zip(
                                self.categories,
                                catscores.tolist(),
                                catlogits.tolist(),
                            )
                        },
                    },
                )
            ]
            for score, classIdx, logit, catscores, catlogits in zip(
                scores.max(axis=1).tolist(),
                scores.argmax(axis=1).tolist(),
                logits.numpy().max(axis=1).tolist(),
                scores,
                logits.numpy(),
            )
        ]

    def compute_targets(self, label_groups):
        raise NotImplementedError("CLIP traning is not supported.")


class CLIPNearestNeighbors(Classifier):
    """A wrapper for OpenAI's CLIP classifier."""

    def __init__(
        self,
        categories,
        model_name="RN50",
        device="cpu",
    ):
        self.model = ModifiedCLIP(model_name, device=device)
        self.resize_config = {
            "method": "fit",
            "width": self.model.model.visual.input_resolution,
            "height": self.model.model.visual.input_resolution,
        }
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(**DEFAULT_NORMALIZATION),
            ]
        )
        self.set_device(device)
        self.model_name = model_name
        self.categories = mira.core.Categories.from_categories(categories)

    def compute_inputs(self, images: np.ndarray):
        return self.preprocess(super().compute_inputs(images))

    def invert_targets(self, y):
        if self.model.exemplar_vectors is None or self.model.exemplar_mapping is None:
            raise ValueError("This model has not been trained.")
        y = y / y.norm(dim=1, keepdim=True)
        logits = (
            (
                self.model.logit_scale.exp()  # type: ignore
                * y
                @ (
                    self.model.exemplar_vectors
                    / self.model.exemplar_vectors.norm(dim=1, keepdim=True)  # type: ignore
                ).t()
            )
            .detach()
            .cpu()
            .numpy()
        )
        scores = torch.softmax(torch.Tensor(logits), dim=-1).numpy()
        scores, logits = [
            np.concatenate(
                [
                    raw[:, indices].sum(axis=1, keepdims=True)
                    for indices in self.model.exemplar_mapping.cpu()  # type: ignore
                    .numpy()
                    .astype("bool")
                ],
                axis=1,
            )
            for raw in [scores, logits]
        ]
        return [
            [
                mira.core.Label(
                    category=self.categories[classIdx],
                    score=score,
                    metadata={
                        "logit": logit,
                        "embedding": yi,
                        "raw": {
                            category.name: {"score": score, "logit": logit}
                            for category, score, logit in zip(
                                self.categories,
                                catscores.tolist(),
                                catlogits.tolist(),
                            )
                        },
                    },
                )
            ]
            for score, classIdx, logit, catscores, catlogits, yi in zip(
                scores.max(axis=1).tolist(),
                scores.argmax(axis=1).tolist(),
                logits.max(axis=1).tolist(),
                scores,
                logits,
                y,
            )
        ]

    def compute_targets(self, label_groups):
        raise NotImplementedError("CLIP traning is not supported.")

    def fit(self, training, batch_size: int = 8, progress=False):
        """Set the exemplar vectors for this model."""
        self.model.exemplar_vectors = torch.cat(
            self.batch_inference(
                items=training,
                process=lambda batch: [self.model(self.compute_inputs(batch.images))],
                batch_size=batch_size,
                progress=progress,
            )[1]
        )
        self.model.exemplar_mapping = torch.Tensor(
            np.array(
                [
                    [any(c.name == l.category.name for l in e.labels) for e in training]
                    for c in self.categories
                ]
            ),
            device=self.device,
        )
