from abc import ABC, abstractmethod
import typing
import itertools
import random

import tensorflow as tf
import numpy as np

from .. import metrics
from ..core import Scene, SceneCollection, Annotation, AnnotationConfiguration, utils


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
        """Compute a list of annotation groups from model output."""

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
        annotation_groups: typing.List[typing.List[Annotation]],
        input_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            annotation_groups: A list of lists of annotation groups.
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
            self.compute_targets(collection.annotation_groups, images[0].shape),
        )

    def batch_generator(
        self,
        collection: typing.Union[SceneCollection, str],
        train_shape: typing.Tuple[int, int, int],
        batch_size: int = 1,
        augmenter: utils.AugmenterProtocol = None,
        shuffle=True,
    ):
        """Create a batch generator from a collection or TFRecords pattern."""
        if isinstance(collection, SceneCollection):
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
                yield self.compute_Xy(sample)
        elif isinstance(collection, str):
            dataset = tf.data.Dataset.list_files(collection).interleave(
                tf.data.TFRecordDataset
            )
            if shuffle:
                dataset = dataset.shuffle(64)
            dataset = dataset.prefetch(batch_size).batch(batch_size).repeat()
            for records in dataset:
                sample = (
                    SceneCollection(
                        scenes=[
                            Scene.from_example(
                                record, annotation_config=self.annotation_config
                            )
                            for record in records
                        ],
                        annotation_config=self.annotation_config,
                    )
                    .augment(augmenter=augmenter)
                    .fit(height=train_shape[0], width=train_shape[1])[0]
                )
                yield self.compute_Xy(collection=sample)
        else:
            raise NotImplementedError(f"Unknown collection type: {type(collection)}")

    def dataset_from_collection(
        self,
        collection: typing.Union[SceneCollection, str],
        batch_size: int,
        augmenter: utils.AugmenterProtocol = None,
        train_shape: typing.Tuple[int, int, int] = None,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """Create a tf.Dataset generating targets for this
        detector from a scene collection or TFRecords string."""
        if isinstance(collection, SceneCollection):
            assert (
                collection.annotation_config == self.annotation_config
            ), "The collection configuration clashes with detector"
            assert (
                collection.consistent
            ), "The collection has inconsistent annotation configuration"
        if train_shape is None:
            train_shape = getattr(self, "training_model", self.model).input_shape[1:]
            assert all(
                s is not None for s in train_shape
            ), "train_shape must be provided for this model."

        batch_generator = self.batch_generator(
            collection=collection,
            batch_size=batch_size,
            augmenter=augmenter,
            shuffle=shuffle,
            train_shape=train_shape,
        )
        output_signature = tuple(
            tensorspec_from_data(data) for data in next(batch_generator)
        )
        return tf.data.Dataset.from_generator(
            lambda: batch_generator,
            output_signature=output_signature,
        )

    def train(
        self,
        training: typing.Union[SceneCollection, str],
        validation: typing.Union[SceneCollection, str] = None,
        batch_size: int = 1,
        augmenter: utils.AugmenterProtocol = None,
        train_shape: typing.Tuple[int, int, int] = None,
        augment_validation: bool = False,
        **kwargs,
    ):
        """Run training job. All additional keyword arguments
        passed to Keras' `fit_generator`.

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
        training_model = getattr(self, "training_model", self.model)

        if "steps_per_epoch" not in kwargs:
            assert not isinstance(
                training, str
            ), "You must set steps_per_epoch if a dataset pattern is used."
            kwargs["steps_per_epoch"] = int(len(training) // batch_size)
        if validation is not None and "validation_steps" not in kwargs:
            assert not isinstance(
                validation, str
            ), "You must set validation_steps if a dataset pattern is used."
            kwargs["validation_steps"] = int(len(validation) // batch_size)
        if "epochs" not in kwargs:
            kwargs["epochs"] = 1000
        training_dataset = self.dataset_from_collection(
            training,
            batch_size=batch_size,
            augmenter=augmenter,
            train_shape=train_shape,
            shuffle=True,
        )
        if validation is not None:
            kwargs["validation_data"] = self.dataset_from_collection(
                validation,
                batch_size=batch_size,
                augmenter=augmenter if augment_validation else None,
                train_shape=train_shape,
                shuffle=False,
            )
        return training_model.fit(training_dataset, **kwargs)

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
