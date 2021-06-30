from abc import ABC, abstractmethod
import types
import typing

import torch
import tqdm
import numpy as np
import timm.optim
import timm.scheduler

from .. import metrics
from ..core import SceneCollection, Annotation, AnnotationConfiguration, utils


class Detector(ABC):
    """Abstract base class for a detector."""

    model: torch.nn.Module
    backbone: torch.nn.Module
    annotation_config: AnnotationConfiguration
    training_model: typing.Optional[torch.nn.Module]

    @abstractmethod
    def invert_targets(
        self,
        y: typing.List[np.ndarray],
        input_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
        threshold: float = 0.5,
        **kwargs,
    ) -> typing.List[typing.List[Annotation]]:
        """Compute a list of annotation groups from model output."""

    def detect(self, image: np.ndarray, **kwargs) -> typing.List[Annotation]:
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
    ) -> typing.List[typing.List[Annotation]]:
        """
        Perform object detection on a batch of images.

        Args:
            images: A list of images
            threshold: The detection threshold for the images
            batch_size: The batch size to use with the underlying model

        Returns:
            A list of lists of annotations.
        """
        images, scales = list(
            zip(*[self._scale_to_model_size(image) for image in images])
        )
        annotation_groups = []
        self.model.eval()
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

    def _scale_to_model_size(self, image: np.ndarray):
        height, width = self.input_shape
        image, scale = utils.fit(image=image, width=width, height=height)
        return image, scale

    @abstractmethod
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

    @abstractmethod
    def compute_targets(
        self,
        annotation_groups: typing.List[typing.List[Annotation]],
        input_shape: typing.Union[
            typing.Tuple[int, int], typing.Tuple[int, int, int]
        ] = None,
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            annotation_groups: A list of lists of annotation groups.
            input_shape: The assumed image input shape.

        Returns:
            The output(s) that will be used by detector.train()
        """

    def freeze_backbone(self):
        """Freeze the body of the model, leaving the final classification and
        regression layer as trainable."""
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        for m in self.model.backbone.modules():
            m.eval()

    def unfreeze_backbone(self):
        """Unfreeze the body of the model, making all layers trainable."""
        for p in self.model.backbone.parameters():
            p.requires_grad = True
        for m in self.model.backbone.modules():
            m.train()

    def compute_Xy(self, collection: SceneCollection):
        """Compute the X, y  representation for a collection."""
        images = collection.images
        return (
            self.compute_inputs(images),
            self.compute_targets(collection.annotation_groups, images[0].shape),
        )

    def train(
        self,
        training: typing.Union[SceneCollection, str],
        validation: typing.Union[SceneCollection, str] = None,
        batch_size: int = 1,
        augmenter: utils.AugmenterProtocol = None,
        epochs=100,
    ):
        """Run training job.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            batch_size: The batch size to use for training
            augmenter: The augmenter for generating samples
            train_shape: The shape to use for training the model
                (assuming the model does not have fixed input
                size).
            augment_validation: Whether to apply augmentation
                to the validation set.
        """
        optimizer = timm.optim.create_optimizer_v2(
            self.training_model, learning_rate=1e-2, weight_decay=4e-5
        )
        scheduler, num_epochs = timm.scheduler.create_scheduler(
            types.SimpleNamespace(
                sched="cosine",
                epochs=epochs,
                min_lr=1e-5,
                decay_rate=0.1,
                warmup_lr=1e-4,
                warmup_epochs=5,
                cooldown_epochs=10,
            ),
            optimizer=optimizer,
        )
        if validation is not None:
            validation = validation.fit(
                width=self.input_shape[1], height=self.input_shape[0]
            )[0]
        for epoch in range(num_epochs):
            with tqdm.trange(len(training) // batch_size) as t:
                self.training_model.train()
                t.set_description(f"Epoch {epoch + 1} / {epochs}")
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    batch = training.assign(
                        scenes=[
                            training[idx] for idx in range(start, start + batch_size)
                        ]
                    )
                    if augmenter is not None:
                        batch = batch.augment(augmenter=augmenter)
                    batch = batch.fit(
                        width=self.input_shape[1], height=self.input_shape[0]
                    )[0]
                    optimizer.zero_grad()
                    loss = self.training_model(
                        self.compute_inputs(batch.images),
                        self.compute_targets(annotation_groups=batch.annotation_groups),
                    )["loss"]
                    loss.backward()
                    cum_loss += loss.detach().numpy()
                    avg_loss = cum_loss / (batchIdx + 1)
                    optimizer.step()
                    scheduler.step(epoch)
                    t.set_postfix(loss=avg_loss)
                    t.update()
                if validation is not None:
                    t.set_postfix(
                        mAP={
                            k: round(v, 2)
                            for k, v in metrics.mAP(
                                true_collection=validation,
                                pred_collection=validation.assign(
                                    scenes=[
                                        scene.assign(annotations=annotations)
                                        for scene, annotations in zip(
                                            validation,
                                            self.detect_batch(
                                                images=validation.images, threshold=0.01
                                            ),
                                        )
                                    ]
                                ),
                            ).items()
                        },
                        loss=avg_loss,
                    )

    def mAP(self, collection: SceneCollection, iou_threshold=0.5):
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
                s.assign(annotations=self.detect(s.image, threshold=0.05))
                for s in collection
            ]
        )
        return metrics.mAP(
            true_collection=collection,
            pred_collection=pred,
            iou_threshold=iou_threshold,
        )
