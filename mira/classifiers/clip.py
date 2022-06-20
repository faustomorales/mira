import typing
import math

import typing_extensions as tx
import mira.core
import mira.thirdparty.clip.clip as clip

import tqdm
import torch
import torchvision
import numpy as np

SimplePrediction = tx.TypedDict("SimplePrediction", {"logit": float, "score": float})
ClassifierPrediction = tx.TypedDict(
    "ClassifierPrediction",
    {
        "score": float,
        "logit": float,
        "label": str,
        "raw": typing.Dict[str, SimplePrediction],
    },
)


class CLIP:
    """A wrapper for OpenAI's CLIP classifier."""

    def __init__(self, annotation_config, model_name="RN50", device="cpu"):
        self.model = clip.load(model_name, device=device)[0]
        px = self.model.visual.input_resolution
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    px, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                ),
                torchvision.transforms.CenterCrop(px),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.device = device
        self.model_name = model_name
        self.annotation_config = annotation_config
        self.text_vectors = self.model.encode_text(
            clip.tokenize([c.name for c in self.annotation_config]).to(self.device)
        )

    def classify(
        self,
        images: typing.List[typing.Union[str, np.ndarray]],
        batch_size=32,
        progress=False,
    ) -> typing.List[ClassifierPrediction]:
        """Classify a batch of images."""
        predictions: typing.List[ClassifierPrediction] = []
        iterator = range(0, len(images), batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator, total=math.ceil(len(images) / batch_size))
        for start in iterator:
            with torch.no_grad():
                image_vectors = self.model.encode_image(
                    torch.stack(
                        [
                            self.preprocess(
                                torch.Tensor(
                                    (
                                        image
                                        if isinstance(image, np.ndarray)
                                        else mira.core.utils.read(image)
                                    ).transpose(2, 0, 1)
                                    / 255.0
                                ).to(self.device)
                            )
                            for image in images[start : start + batch_size]
                        ]
                    )
                )
                logits = (
                    (
                        self.model.logit_scale.exp()
                        * (image_vectors / image_vectors.norm(dim=1, keepdim=True))
                        @ (
                            self.text_vectors
                            / self.text_vectors.norm(dim=1, keepdim=True)
                        ).t()
                    )
                    .detach()
                    .cpu()
                )
                scores = logits.softmax(dim=-1).numpy()
                predictions.extend(
                    [
                        {
                            "label": self.annotation_config[classIdx].name,
                            "score": score,
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
                )
        return predictions
