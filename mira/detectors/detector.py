from abc import ABC, abstractmethod
import typing
import itertools
import random

import tensorflow as tf
import imgaug as ia
import numpy as np

from .. import metrics
from ..core import SceneCollection, Annotation, AnnotationConfiguration, utils


def tensorspec_from_data(data: typing.Union[dict, np.ndarray, list, tuple]):
    """Build a tf.TensorSpec from a numpy array or dict of
    key/numpy arrays."""
    if isinstance(data, dict):
        return {
            k: tf.TensorSpec(shape=(None, *v.shape[1:]), dtype=v.dtype)
            for k, v in data.items()
        }
    if isinstance(data, np.ndarray):
        return tf.TensorSpec(shape=(None, *data.shape[1:]), dtype=data.dtype)
    if isinstance(data, (list, tuple)):
        return tuple(tensorspec_from_data(d) for d in data)
    raise NotImplementedError(f"Cannot convert {type(data)} to TensorSpec.")


class Detector(ABC):
    """Abstract base class for a detector."""

    model: tf.keras.models.Model
    backbone: tf.keras.models.Model
    annotation_config: AnnotationConfiguration
    training_model: typing.Optional[tf.keras.models.Model]

    @abstractmethod
    def compile(self):
        """Compile the model using known configuration for loss and optimizer."""

    @abstractmethod
    def invert_targets(
        self,
        y: typing.List[np.ndarray],
        input_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
        threshold: float = 0.5,
        **kwargs,
    ) -> typing.List[typing.List[Annotation]]:
        """Compute a scene collection from model output."""

    def detect(
        self, image: np.ndarray, threshold: float = 0.5, **kwargs
    ) -> typing.List[Annotation]:
        """Run detection for a given image.

        Args:
            image: The image to run detection on

        Returns:
            A list of annotations
        """
        image, scale = self._scale_to_model_size(image)
        X = self.compute_inputs([image])
        y = self.model.predict(X)
        annotations = self.invert_targets(
            y, input_shape=image.shape, threshold=threshold, **kwargs
        )[0]
        return [a.resize(1 / scale) for a in annotations]

    def detect_batch(
        self,
        images: typing.List[np.ndarray],
        threshold: float = 0.5,
        batch_size: int = 32,
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
        X = self.compute_inputs(images)
        y = self.model.predict(X, batch_size=batch_size)
        annotation_groups = self.invert_targets(
            y, input_shape=X.shape[1:], threshold=threshold
        )
        return [
            [a.resize(1 / scale) for a in annotations]
            for annotations, scale in zip(annotation_groups, scales)
        ]

    def _scale_to_model_size(self, image: np.ndarray):
        height, width = self.model.input_shape[1:3]
        if height is not None and width is not None:
            image, scale = utils.fit(image=image, width=width, height=height)
        else:
            scale = 1
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
        collection: SceneCollection,
        input_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            collection: The scene collection for which the
                outputs should be calculated.
            input_shape: The assumed image input shape.

        Returns:
            The output(s) as a list of numpy arrays
        """

    def freeze_backbone(self):
        """Freeze the body of the model, leaving the final classification and
        regression layer as trainable."""
        for l in self.backbone.layers:
            l.trainable = False
        self.compile()

    def unfreeze_backbone(self):
        """Unfreeze the body of the model, making all layers trainable."""
        for l in self.backbone.layers:
            l.trainable = True
        self.compile()

    def compute_Xy(self, collection: SceneCollection):
        """Compute the X, y  representation for a collection."""
        images = collection.images
        return (
            self.compute_inputs(images),
            self.compute_targets(collection, images[0].shape),
        )

    def batch_generator(
        self,
        collection: SceneCollection,
        train_shape: typing.Tuple[int, int, int],
        batch_size: int = 1,
        augmenter: ia.augmenters.Augmenter = None,
        shuffle=True,
    ):
        """Create a batch generator from a collection."""
        index = np.arange(len(collection)).tolist()
        for idx in itertools.cycle(range(0, len(collection), batch_size)):
            if idx == 0 and shuffle:
                random.shuffle(index)
            sample = (
                collection.assign(
                    scenes=[collection[i] for i in index[idx : idx + batch_size]]
                )
                .augment(augmenter=augmenter)
                .fit(height=train_shape[0], width=train_shape[1])[0]
            )
            X, y = self.compute_Xy(sample)
            yield X, y

    def train(
        self,
        training: SceneCollection,
        validation: SceneCollection = None,
        batch_size: int = 1,
        augmenter: ia.augmenters.Augmenter = None,
        train_shape: typing.Tuple[int, int, int] = None,
        **kwargs,
    ):
        """Run training job. All additional keyword arguments
        passed to Keras' `fit_generator`.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            batch_size: The batch size to use for training
            augmenter: The augmenter for generating samples
        """
        assert (
            training.annotation_config == self.annotation_config
        ), "The training set configuration clashes with detector"
        assert (
            validation is None or validation.annotation_config == self.annotation_config
        ), "The validation set configuration clashes with detector"
        assert (
            training.consistent
        ), "The training set has inconsistent annotation configuration"
        assert (
            validation is None or validation.consistent
        ), "The validation set has inconsistent annotation configuration"
        if hasattr(self, "training_model"):
            assert self.training_model is not None
            training_model = self.training_model  # pylint: disable=no-member
        else:
            training_model = self.model
        if train_shape is None:
            train_shape = training_model.input_shape[1:]
            assert all(
                s is not None for s in train_shape
            ), "train_shape must be provided for this model."
        if "steps_per_epoch" not in kwargs:
            kwargs["steps_per_epoch"] = int(len(training) // batch_size)
        if validation is not None and "validation_steps" not in kwargs:
            kwargs["validation_steps"] = int(len(validation) // batch_size)
        if "epochs" not in kwargs:
            kwargs["epochs"] = 1000
        training_generator = self.batch_generator(
            collection=training,
            batch_size=batch_size,
            augmenter=augmenter,
            shuffle=True,
            train_shape=train_shape,
        )
        output_signature = tuple(
            tensorspec_from_data(data) for data in next(training_generator)
        )
        training_dataset = tf.data.Dataset.from_generator(
            lambda: training_generator,
            output_signature=output_signature,
        )
        if validation is None:
            history = training_model.fit(training_dataset, **kwargs)
        else:
            validation_dataset = tf.data.Dataset.from_generator(
                lambda: self.batch_generator(
                    collection=validation,
                    batch_size=batch_size,
                    augmenter=None,
                    train_shape=train_shape,
                    shuffle=False,
                ),
                output_signature=output_signature,
            )
            kwargs["validation_steps"] = int(len(validation) // batch_size)
            history = training_model.fit(
                training_dataset,
                validation_data=validation_dataset,
                **kwargs,
            )
        return history

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
                s.assign(annotations=self.detect(s.image, threshold=0))
                for s in collection
            ]
        )
        return metrics.mAP(
            true_collection=collection,
            pred_collection=pred,
            iou_threshold=iou_threshold,
        )
