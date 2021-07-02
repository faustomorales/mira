import abc
import types
import typing
import random

import torch
import tqdm
import numpy as np
import timm.optim
import timm.scheduler
import typing_extensions as tx

from .. import metrics as mm
from .. import core as mc

DEFAULT_SCHEDULER_PARAMS = dict(
    sched="cosine",
    min_lr=1e-5,
    decay_rate=0.1,
    warmup_lr=1e-4,
    warmup_epochs=5,
    cooldown_epochs=10,
)

DEFAULT_OPTIMIZER_PARAMS = dict(learning_rate=1e-2, weight_decay=4e-5)


def _loss_from_loss_dict(loss_dict: typing.Dict[str, torch.Tensor]):
    if "loss" in loss_dict:
        return loss_dict["loss"]
    return sum(loss for loss in loss_dict.values())


# pylint: disable=too-few-public-methods
class CallbackProtocol(tx.Protocol):
    """A protocol defining how we expect callbacks to
    behave."""

    def __call__(
        self, detector: "Detector", summaries: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.Dict[str, typing.Any]:
        pass


class Detector(abc.ABC):
    """Abstract base class for a detector."""

    model: torch.nn.Module
    backbone: torch.nn.Module
    annotation_config: mc.AnnotationConfiguration
    training_model: typing.Optional[torch.nn.Module]

    @abc.abstractmethod
    def invert_targets(
        self,
        y: typing.Any,
        threshold: float = 0.5,
        **kwargs,
    ) -> typing.List[typing.List[mc.Annotation]]:
        """Compute a list of annotation groups from model output."""

    def detect(self, image: np.ndarray, **kwargs) -> typing.List[mc.Annotation]:
        """Run detection for a given image. All other args passed to invert_targets()

        Args:
            image: The image to run detection on

        Returns:
            A list of annotations
        """
        self.model.eval()
        image, scale = self._scale_to_model_size(image)
        with torch.no_grad():
            annotations = self.invert_targets(
                self.model(self.compute_inputs([image])), **kwargs
            )[0]
        return [a.resize(1 / scale) for a in annotations]

    def detect_batch(
        self,
        images: typing.List[np.ndarray],
        batch_size: int = 32,
        **kwargs,
    ) -> typing.List[typing.List[mc.Annotation]]:
        """
        Perform object detection on a batch of images.

        Args:
            images: A list of images
            threshold: The detection threshold for the images
            batch_size: The batch size to use with the underlying model

        Returns:
            A list of lists of annotations.
        """
        self.model.eval()
        images, scales = list(
            zip(*[self._scale_to_model_size(image) for image in images])
        )
        annotation_groups = []
        for start in range(0, len(images), batch_size):
            annotation_groups.extend(
                self.invert_targets(
                    self.model(
                        self.compute_inputs(images[start : start + batch_size]),
                    ),
                    **kwargs,
                )
            )
        return [
            [a.resize(1 / scale) for a in annotations]
            for annotations, scale in zip(annotation_groups, scales)
        ]

    @property
    @abc.abstractmethod
    def input_shape(self) -> typing.Tuple[int, int, int]:
        """Obtain the input shape for this model."""

    @abc.abstractmethod
    def set_input_shape(self, width: int, height: int):
        """Set the input shape for this model."""

    def _scale_to_model_size(self, image: np.ndarray):
        height, width = self.input_shape[:2]
        image, scale = mc.utils.fit(image=image, width=width, height=height)
        return image, scale

    @abc.abstractmethod
    def compute_inputs(self, images: typing.List[np.ndarray]) -> np.ndarray:
        """Convert images into suitable model inputs. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            images: The images to convert

        Returns:
            The input to the model
        """

    @abc.abstractmethod
    def compute_targets(
        self,
        annotation_groups: typing.List[typing.List[mc.Annotation]],
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            annotation_groups: A list of lists of annotation groups.

        Returns:
            The output(s) that will be used by detector.train()
        """

    def freeze_backbone(self):
        """Freeze the body of the model, leaving the final classification and
        regression layer as trainable."""
        for p in self.model.backbone.parameters():  # type: ignore
            p.requires_grad = False
        for m in self.model.backbone.modules():  # type: ignore
            m.eval()

    def unfreeze_backbone(self):
        """Unfreeze the body of the model, making all layers trainable."""
        for p in self.model.backbone.parameters():  # type: ignore
            p.requires_grad = True
        for m in self.model.backbone.modules():  # type: ignore
            m.train()

    def train(
        self,
        training: mc.SceneCollection,
        validation: mc.SceneCollection = None,
        batch_size: int = 1,
        augmenter: mc.utils.AugmenterProtocol = None,
        train_backbone: bool = True,
        epochs=100,
        shuffle=True,
        callbacks: typing.List[CallbackProtocol] = None,
        optimizer_params=None,
        scheduler_params=None,
    ):
        """Run training job.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            batch_size: The batch size to use for training
            augmenter: The augmenter for generating samples
            train_backbone: Whether to fit the backbone.
            epochs: The number of epochs to train.
            shuffle: Whether to shuffle the training data on each epoch.
            callbacks: A list of functions that accept the detector as well
                as a list of previous summaries and returns a dict of summary keys
                and values which will be added to the current summary. It can raise
                StopIteration to stop training early.
            optimizer_params: Passed to timm.optim.create_optimizer_v2 to build
                the optimizer.
            scheduler_params: Passed to timm.schduler.create_scheduler to build
                the scheduler.
        """
        training_model = (
            self.training_model if hasattr(self, "training_modle") else self.model
        )
        assert training_model is not None
        optimizer = timm.optim.create_optimizer_v2(
            training_model, **(optimizer_params or DEFAULT_OPTIMIZER_PARAMS)
        )
        scheduler, num_epochs = timm.scheduler.create_scheduler(
            types.SimpleNamespace(
                **{**(scheduler_params or DEFAULT_SCHEDULER_PARAMS), "epochs": epochs}
            ),
            optimizer=optimizer,
        )
        if validation is not None:
            validation = validation.fit(
                width=self.input_shape[1], height=self.input_shape[0]
            )[0]
        train_index = np.arange(len(training)).tolist()
        summaries = []
        for epoch in range(num_epochs):
            with tqdm.trange(len(training) // batch_size) as t:
                training_model.train()
                if not train_backbone:
                    self.freeze_backbone()
                else:
                    self.unfreeze_backbone()
                t.set_description(f"Epoch {epoch + 1} / {num_epochs}")
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    batch = training.assign(
                        scenes=[
                            training[train_index[idx]]
                            for idx in range(start, start + batch_size)
                        ]
                    )
                    if augmenter is not None:
                        batch = batch.augment(augmenter=augmenter)
                    batch = batch.fit(
                        width=self.input_shape[1], height=self.input_shape[0]
                    )[0]
                    optimizer.zero_grad()
                    loss = _loss_from_loss_dict(
                        training_model(
                            self.compute_inputs(batch.images),
                            self.compute_targets(
                                annotation_groups=batch.annotation_groups
                            ),
                        )
                    )
                    loss.backward()
                    cum_loss += loss.detach().numpy()
                    avg_loss = cum_loss / (batchIdx + 1)
                    optimizer.step()
                    scheduler.step(epoch)
                    t.set_postfix(loss=avg_loss)
                    t.update()
                summary: typing.Dict[str, typing.Any] = {"loss": avg_loss}
                summaries.append(summary)
                if validation is not None:
                    summary["val_mAP"] = {
                        k: round(v, 2)
                        for k, v in self.mAP(
                            collection=validation, batch_size=batch_size
                        ).items()
                    }
                if callbacks is not None:
                    try:
                        for callback in callbacks:
                            for k, v in callback(
                                detector=self, summaries=summaries
                            ).items():
                                summary[k] = v
                    except StopIteration:
                        return summaries
                t.set_postfix(**summary)
        return summaries

    def mAP(self, collection: mc.SceneCollection, iou_threshold=0.5, batch_size=32):
        """Compute the mAP metric for a given collection
        of ground truth scenes.

        Args:
            collection: The collection to evaluate
            iou_threshold: The IoU threshold required for
                a match

        Returns:
            mAP score
        """
        pred = collection.assign(
            scenes=[
                scene.assign(annotations=annotations)
                for scene, annotations in zip(
                    collection,
                    self.detect_batch(
                        images=collection.images, threshold=0.01, batch_size=batch_size
                    ),
                )
            ]
        )
        return mm.mAP(
            true_collection=collection,
            pred_collection=pred,
            iou_threshold=iou_threshold,
        )
