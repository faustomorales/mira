from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from imgaug import augmenters as iaa
import numpy as np

from .. import metrics
from ..core import (
    SceneCollection,
    Annotation,
    Image
)


class Detector(ABC):
    def detect(
        self,
        image: Union[Image, np.ndarray],
        threshold: float=0.5
    ) -> List[Annotation]:
        """Run detection for a given image.

        Args:
            image: The image to run detection on

        Returns:
            A list of annotations
        """
        image = image.view(Image)
        image, scale = self._scale_to_model_size(image)
        X = self.compute_inputs([image])
        y = self.model.predict(X)
        annotations = self.invert_targets(
            y,
            images=[image],
            threshold=threshold
        )[0].annotations
        return [a.resize(1/scale) for a in annotations]

    def detect_batch(
        self,
        images: List[Union[Image, np.ndarray]],
        threshold: float=0.5,
        batch_size: int=32
    ) -> List[List[Annotation]]:
        """
        Perform object detection on a batch of images.

        Args:
            images: A list of images
            threshold: The detection threshold for the images
            batch_size: The batch size to use with the underlying model

        Returns:
            A list of lists of annotations.
        """
        images = [image.view(Image) for image in images]
        images, scales = list(
            zip(*[self._scale_to_model_size(image) for image in images])
        )
        X = self.compute_inputs(images)
        y = self.model.predict(X, batch_size=batch_size)
        scenes = self.invert_targets(
            y,
            images=images,
            threshold=threshold
        )
        return [
            [
                a.resize(1/scale) for a in scene.annotations
            ] for scene, scale in zip(scenes, scales)
        ]

    def _scale_to_model_size(self, image: Image):
        height, width = self.model.input_shape[1:3]
        if height is not None and width is not None:
            image, scale = image.fit(
                width=width,
                height=height
            )
        else:
            scale = 1
        return image, scale

    @abstractmethod
    def compute_inputs(self, images: List[Image]) -> np.ndarray:
        """Convert images into suitable model inputs. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            images: The images to convert

        Returns:
            The input to the model
        """
        pass

    @abstractmethod
    def compute_targets(
        self,
        collection: SceneCollection
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            collection: The scene collection for which the
                outputs should be calculated.

        Returns:
            The output(s) as a list of numpy arrays
        """
        pass

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
        return (
            self.compute_inputs(collection.images),
            self.compute_targets(collection)
        )

    def batch_generator(
        self,
        collection: SceneCollection,
        train_shape: Tuple[int, int, int],
        batch_size: int=1,
        augmenter: iaa.Augmenter=None
    ):
        while True:
            sample = collection.sample(
                n=batch_size
            ).augment(
                augmenter=augmenter
            ).fit(
                height=train_shape[0],
                width=train_shape[1]
            )[0]
            yield self.compute_Xy(sample)

    def train(
        self,
        training: SceneCollection,
        validation: SceneCollection=None,
        batch_size: int=1,
        augmenter: iaa.Augmenter=None,
        train_shape: Tuple[int, int, int]=None,
        **kwargs
    ):
        """Run training job. All additional keyword arguments
        passed to Keras' `fit_generator`. Note that, at a minimum,
        you must provide the `epochs` and `steps_per_epoch` arguments.
        If validation data is provided, you will also need to provide a
        `validation_steps` argument.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            batch_size: The batch size to use for training
            augmenter: The augmenter for generating samples
        """
        assert training.annotation_config == self.annotation_config, \
            'The training set configuration clashes with detector'
        assert (
            validation is None or
            validation.annotation_config == self.annotation_config
        ), \
            'The validation set configuration clashes with detector'
        assert training.consistent, \
            'The training set has inconsistent annotation configuration'
        assert (
            validation is None or
            validation.consistent
        ), \
            'The validation set has inconsistent annotation configuration'
        if hasattr(self, 'training_model'):
            training_model = self.training_model
        else:
            training_model = self.model
        if train_shape is None:
            train_shape = training_model.input_shape[1:]
            assert all(s is not None for s in train_shape), \
                'train_shape must be provided for this model.'
        training_generator = self.batch_generator(
            collection=training,
            batch_size=batch_size,
            augmenter=augmenter,
            train_shape=train_shape
        )
        if validation is None:
            history = training_model.fit_generator(
                generator=training_generator,
                **kwargs
            )
        else:
            validation = validation.fit(
                height=train_shape[0],
                width=train_shape[1]
            )[0]
            history = training_model.fit_generator(
                generator=training_generator,
                validation_data=self.compute_Xy(validation),
                **kwargs
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
                s.assign(
                    annotations=self.detect(s.image, threshold=0)
                ) for s in collection
            ]
        )
        return metrics.mAP(
            true_collection=collection,
            pred_collection=pred,
            iou_threshold=iou_threshold
        )