import abc
import math
import typing

import tqdm
import torch
import numpy as np
import typing_extensions as tx

from .. import core as mc

SimplePrediction = tx.TypedDict("SimplePrediction", {"logit": float, "score": float})
ClassifierPrediction = tx.TypedDict(
    "ClassifierPrediction",
    {
        "label": mc.Label,
        "logit": float,
        "raw": typing.Dict[str, SimplePrediction],
    },
)
TrainSplitState = tx.TypedDict(
    "TrainSplitState", {"true": typing.List[int], "pred": typing.List[int]}
)
TrainState = tx.TypedDict(
    "TrainState", {"train": TrainSplitState, "val": TrainSplitState}
)


class Classifier(mc.torchtools.BaseModel):
    """Abstract base class for classifier."""

    @abc.abstractmethod
    def invert_targets(self, y: typing.Any) -> typing.List[ClassifierPrediction]:
        """Convert model outputs back into predictions."""

    @abc.abstractmethod
    def compute_targets(
        self,
        label_groups: typing.List[typing.List[mc.annotation.Category]],
    ):
        """Compute the targets for a batch of labeled images."""

    def classify(
        self,
        images: typing.List[typing.Union[str, np.ndarray]],
        batch_size=32,
        progress=False,
    ) -> typing.List[ClassifierPrediction]:
        """Classify a batch of images."""
        self.model.eval()
        predictions: typing.List[ClassifierPrediction] = []
        iterator = range(0, len(images), batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator, total=math.ceil(len(images) / batch_size))
        for start in iterator:
            with torch.no_grad():
                current_images, _ = self.resize_to_model_size(
                    [
                        mc.utils.read(image)
                        for image in images[start : start + batch_size]
                    ]
                )
                predictions.extend(
                    self.invert_targets(self.model(self.compute_inputs(current_images)))
                )
        return predictions

    def train(
        self,
        training: mc.SceneCollection,
        validation: mc.SceneCollection = None,
        augmenter: mc.augmentations.AugmenterProtocol = None,
        train_backbone: bool = True,
        train_backbone_bn: bool = True,
        validation_transforms: np.ndarray = None,
        **kwargs,
    ):
        """Run training job. All other arguments passed to mira.core.training.train.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            augmenter: The augmenter for generating samples
            train_backbone: Whether to fit the backbone.
            train_backbone_bn: Whether to fit backbone batchnorm layers.
            callbacks: A list of training callbacks.
            data_dir_prefix: Prefix for the intermediate model artifacts directory.
            validation_transforms: A list of transforms for the images in the validation
                set. If not provided, we assume the identity transform.
        """
        state: TrainState = {
            "train": {"true": [], "pred": []},
            "val": {"true": [], "pred": []},
        }

        def loss(items: typing.List[mc.torchtools.TrainItem]) -> torch.Tensor:
            batch = training.assign(scenes=[i.scene for i in items])
            y = self.model(
                self.compute_inputs(self.resize_to_model_size(batch.images)[0]),
                self.compute_targets(batch.label_groups),
            )
            predictions = self.invert_targets(y)
            state[items[0].split]["true"].extend(
                [self.categories.index(s.labels[0].category) for s in batch]
            )
            state[items[0].split]["pred"].extend(
                [self.categories.index(p["label"].category) for p in predictions]
            )
            return y["loss"]

        def augment(items: typing.List[mc.torchtools.TrainItem]):
            if not augmenter:
                return items
            return [
                mc.torchtools.TrainItem(
                    split=base.split,
                    index=base.index,
                    scene=scene,
                    transform=np.matmul(transform, base.transform),
                )
                for (scene, transform), base in zip(
                    [i.scene.augment(augmenter) for i in items], items
                )
            ]

        def on_epoch_end(summaries: typing.List[dict]):
            summary: typing.Dict[str, typing.Any] = summaries[-1]
            for split, data in zip(["train", "val"], [state["train"], state["val"]]):
                true = np.array(data["true"])
                pred = np.array(data["pred"])
                for idx, category in enumerate(self.categories):
                    tp = ((true == idx) & (pred == idx)).sum()
                    fp = ((true != idx) & (pred == idx)).sum()
                    fn = ((true == idx) & (pred != idx)).sum()
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    summary[f"{split}_{category.name}_f1"] = (
                        2 * precision * recall
                    ) / (precision + recall)
                    summary[f"{split}_{category.name}_recall"] = tp / (tp + fn)
                    summary[f"{split}_{category.name}_precision"] = tp / (tp + fp)
            state["train"] = {"true": [], "pred": []}
            state["val"] = {"true": [], "pred": []}
            return summary

        def on_epoch_start():
            if train_backbone:
                self.unfreeze_backbone(batchnorm=train_backbone_bn)
            else:
                self.freeze_backbone()

        mc.torchtools.train(
            model=self.model,
            training=[
                mc.torchtools.TrainItem(
                    split="train", index=index, transform=np.eye(3), scene=scene
                )
                for index, scene in enumerate(training)
            ],
            validation=[
                mc.torchtools.TrainItem(
                    split="val", index=index, transform=transform, scene=scene
                )
                for index, (scene, transform) in enumerate(
                    zip(
                        validation or [],
                        (
                            validation_transforms
                            or np.eye(3, 3)[np.newaxis].repeat(len(validation), axis=0)
                        )
                        if validation
                        else [],
                    )
                )
            ],
            loss=loss,
            augment=augment,
            on_epoch_start=on_epoch_start,
            on_epoch_end=on_epoch_end,
            **kwargs,
        )
