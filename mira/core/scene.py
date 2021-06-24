"""Scene and SceneCollection objects"""

# pylint: disable=invalid-name,len-as-condition

import os
import json
import typing
import logging
import math
import io

import tensorflow as tf
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import validators

from .annotation import AnnotationConfiguration, Annotation
from .selection import Selection
from . import utils

log = logging.getLogger(__name__)


class Scene:
    """A single annotated image.

    Args:
        annotation_config: The configuration for annotations for the
            image.
        annotations: The list of annotations.
        image: The image that was annotated. Can be lazy-loaded by passing
            a string filepath.
        metadata: Metadata about the scene as a dictionary
        cache: Defines caching behavior for the image. If `True`, image is
            loaded into memory the first time that the image is requested.
            If `False`, image is loaded from the file path or URL whenever
            the image is requested.
    """

    def __init__(
        self,
        annotation_config: AnnotationConfiguration,
        annotations: typing.List[Annotation],
        image: typing.Union[np.ndarray, str],
        metadata: dict = None,
        cache: bool = False,
    ):
        assert isinstance(
            image, (np.ndarray, str)
        ), "Image must be string or ndarray, not " + str(type(image))
        self.metadata = metadata
        self._image = image
        self._annotations = annotations
        self._annotation_config = annotation_config
        self.cache = cache

    @property
    def image(self) -> np.ndarray:
        """The image that is being annotated"""
        # Check to see if we have an actual image
        # or just a string
        if not isinstance(self._image, str):
            return self._image

        # Check the cache first if image is a URL
        if (
            validators.url(self._image)
            and isinstance(self.cache, str)
            and os.path.isfile(self.cache)
        ):
            self._image = self.cache

        # Load the image
        log.debug("Reading from %s", self._image)
        image = utils.read(self._image)

        # Check how to handle caching the image
        # for future reads
        if self.cache is True:
            self._image = image
        elif self.cache is False:
            pass
        else:
            raise ValueError("Cannot handle cache parameter: {0}.".format(self.cache))
        return image

    @property
    def annotation_config(self):
        """The annotation configuration"""
        return self._annotation_config

    @property
    def annotations(self) -> typing.List[Annotation]:
        """Get the list of annotations"""
        return self._annotations

    def assign(self, **kwargs) -> "Scene":
        """Get a new scene with only the supplied
        keyword arguments changed."""
        if "annotation_config" in kwargs:
            # We need to change all the categories for annotations
            # to match the new annotation configuration.
            annotations = kwargs.get("annotations", self.annotations)
            annotation_config = kwargs["annotation_config"]
            revised = [
                ann.convert(annotation_config=annotation_config) for ann in annotations
            ]
            revised = [ann for ann in revised if ann is not None]
            removed = len(annotations) - len(revised)
            log.debug(
                "Removed %s annotations when changing annotation configuration.",
                removed,
            )
            kwargs["annotations"] = revised
        # We use the _image instead of image to avoid triggering an
        # unnecessary read of the actual image.
        defaults = {
            "annotation_config": self.annotation_config,
            "annotations": self.annotations,
            "image": self._image,
            "cache": self.cache,
            "metadata": self.metadata,
        }
        kwargs = {**defaults, **kwargs}
        return Scene(**kwargs)

    def to_example(self) -> tf.train.Example:
        """Obtain a tf.Example for the scene."""
        image = self.image
        bboxes = self.bboxes()
        bboxes_scaled = bboxes[:, :4].astype("float32") / np.array(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        ).astype("float32")
        image_bytes = io.BytesIO()
        utils.save(image, image_bytes, extension=".png")
        image_bytes.seek(0)
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[0]])
                    ),
                    "image/width": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[1]])
                    ),
                    "image/encoded": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_bytes.read()])
                    ),
                    "image/format": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=["png".encode()])
                    ),
                    "image/object/bbox/xmin": tf.train.Feature(
                        float_list=tf.train.FloatList(value=bboxes_scaled[:, 0])
                    ),
                    "image/object/bbox/ymin": tf.train.Feature(
                        float_list=tf.train.FloatList(value=bboxes_scaled[:, 1])
                    ),
                    "image/object/bbox/xmax": tf.train.Feature(
                        float_list=tf.train.FloatList(value=bboxes_scaled[:, 2])
                    ),
                    "image/object/bbox/ymax": tf.train.Feature(
                        float_list=tf.train.FloatList(value=bboxes_scaled[:, 3])
                    ),
                    "image/metadata": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[json.dumps(self.metadata or {}).encode()]
                        )
                    ),
                    "image/object/class/text": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[
                                self.annotation_config[idx].name.encode()
                                for idx in bboxes[:, -1].tolist()
                            ]
                        )
                    ),
                }
            )
        )

    @classmethod
    def from_example(cls, serialized, annotation_config: AnnotationConfiguration):
        """Load a scene using a serialized tf.Example representation."""
        deserialized = tf.io.parse_single_example(
            serialized,
            features={
                "image/encoded": tf.io.FixedLenFeature((), tf.string),
                "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
                "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
                "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
                "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
                "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
                "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
                "image/object/class/text": tf.io.VarLenFeature(tf.string),
                "image/metadata": tf.io.FixedLenFeature((), tf.string),
            },
        )
        width, height = (
            deserialized["image/width"].numpy(),
            deserialized["image/height"].numpy(),
        )
        image = tf.io.decode_image(deserialized["image/encoded"]).numpy()
        assert (
            width == image.shape[1] and height == image.shape[0]
        ), "Deserialization failed."
        return cls(
            image=image,
            metadata=json.loads(deserialized["image/metadata"].numpy().decode("utf-8")),
            annotations=[
                Annotation(
                    selection=Selection(
                        [[x1 * width, y1 * height], [x2 * width, y2 * height]]
                    ),
                    category=annotation_config[label.decode("utf-8")],
                )
                for x1, x2, y1, y2, label in zip(
                    *map(
                        lambda s: tf.sparse.to_dense(s, default_value=0),
                        [
                            deserialized["image/object/bbox/xmin"],
                            deserialized["image/object/bbox/xmax"],
                            deserialized["image/object/bbox/ymin"],
                            deserialized["image/object/bbox/ymax"],
                        ],
                    ),
                    tf.sparse.to_dense(
                        deserialized["image/object/class/text"], default_value=""
                    ).numpy(),
                )
            ],
            annotation_config=annotation_config,
        )

    def show(self, *args, **kwargs) -> mpl.axes.Axes:
        """Show an annotated version of the image. All arguments
        passed to `mira.core.utils.show()`.
        """
        return utils.show(self.annotated(), *args, **kwargs)

    def scores(self):
        """Obtain an array containing the confidence
        score for each annotation."""
        return np.array([a.score for a in self.annotations])

    def bboxes(self):
        """Obtain an array of shape (N, 5) where the columns are
        x1, y1, x2, y2, class_index where class_index is determined
        from the annotation configuration."""
        # We reshape in order to avoid indexing problems when
        # there are no annotations.
        return self.annotation_config.bboxes_from_group(self.annotations)

    def fit(self, width, height):
        """Obtain a new scene fitted to the given width and height.

        Args:
            width: The new width
            height: The new height

        Returns:
            The new scene and the scale
        """
        image, scale = utils.fit(self.image, width=width, height=height)
        annotations = [ann.resize(scale=scale) for ann in self.annotations]
        return self.assign(image=image, annotations=annotations), scale

    def annotated(
        self, dpi=72, fontsize="x-large", labels=True, opaque=False, color=(255, 0, 0)
    ) -> np.ndarray:
        """Show annotations on the image itself.

        Args:
            dpi: The resolution for the image
            fontsize: How large to show labels
            labels: Whether or not to show labels
            opaque: Whether to draw annotations filled
                in.
            color: The color to use for annotations.
        """
        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.axis("off")
        img_raw = self.image
        img = img_raw
        for ann in self.annotations:
            img = ann.selection.draw(img, color=color, opaque=opaque)
        utils.show(img, ax=ax)
        if labels:
            for ann in self.annotations:
                x1, y1, _, _ = ann.selection.bbox()
                ax.annotate(
                    s=ann.category.name,
                    xy=(x1, y1),
                    fontsize=fontsize,
                    backgroundcolor=(1, 1, 1, 0.5),
                )
        ax.set_xlim(0, img_raw.shape[1])
        ax.set_ylim(img_raw.shape[0], 0)
        fig.canvas.draw()
        raw = io.BytesIO()
        fig.savefig(
            raw,
            dpi=dpi,
            frameon=True,
            pad_inches=0,
            transparent=False,
            bbox_inches="tight",
        )
        plt.close(fig)
        plt.ion()
        raw.seek(0)
        img = utils.read(raw)
        img = img[:, :, :3]
        raw.close()
        return img

    def resize(self, scale) -> "Scene":
        """Obtain a resized version of the scene.

        Args:
            scale: The scale for the new scene.
        """
        return self.assign(
            image=self.image.resize(scale),
            annotations=[ann.resize(scale) for ann in self.annotations],
        )

    def augment(
        self, augmenter: iaa.Augmenter = None, threshold: float = 0.25
    ) -> "Scene":
        """Obtain an augmented version of the scene using the given augmenter.

        Returns:
            The augmented scene
        """
        if augmenter is None:
            return self
        aug = augmenter.to_deterministic()
        keypoints: typing.List[ia.Keypoint] = []
        keypoints_map = {}
        for i, ann in enumerate(self.annotations):
            current = ann.selection.keypoints()
            startIdx = len(keypoints)
            endIdx = startIdx + len(current)
            keypoints_map[i] = (startIdx, endIdx)
            keypoints.extend(current)
        keypoints = ia.KeypointsOnImage(keypoints, shape=self.image.shape)
        image = aug.augment_images([self.image])[0]
        keypoints = aug.augment_keypoints([keypoints])[0].keypoints
        annotations = []
        for i, ann in enumerate(self.annotations):
            startIdx, endIdx = keypoints_map[i]
            current = keypoints[startIdx:endIdx]
            selection = ann.selection.assign_keypoints(current)
            area = selection.area()
            selection = selection.crop(width=image.shape[1], height=image.shape[0])
            if area == 0 or (selection.area() / area < threshold):
                continue
            annotations.append(ann.assign(selection=selection))
        return self.assign(image=image, annotations=annotations)


class SceneCollection:
    """A collection of scenes.

    Args:
        annotation_config: The configuration that should be used for all
            underlying scenes.
        scenes: The list of scenes.
    """

    def __init__(
        self,
        scenes: typing.List[Scene],
        annotation_config: AnnotationConfiguration = None,
    ):
        assert len(scenes) > 0, "A scene collection must have at least one scene"
        if annotation_config is None:
            annotation_config = scenes[0].annotation_config
        for i, s in enumerate(scenes):
            if s.annotation_config != annotation_config:
                raise ValueError(
                    "Scene {0} of {1} has inconsistent configuration.".format(
                        i + 1, len(scenes)
                    )
                )
        self._annotation_config = annotation_config
        self._scenes = scenes

    def __getitem__(self, key):
        return self.scenes[key]

    def __setitem__(self, key, val):
        if key >= len(self.scenes):
            raise ValueError(
                f"Cannot set scene {key} when collection has length {len(self.scenes)}."
            )
        self.scenes[key] = val

    def __len__(self):
        return len(self._scenes)

    def __iter__(self):
        for scene in self._scenes:
            yield scene

    @property
    def scenes(self):
        """The list of scenes"""
        return self._scenes

    @property
    def annotation_config(self):
        """The annotation configuration"""
        return self._annotation_config

    @property
    def annotation_groups(self):
        """The groups of annotations in the collection."""
        return [s.annotations for s in self.scenes]

    @property
    def uniform(self):
        """Specifies whether all scenes in the collection are
        of the same size. Note: This will trigger an image load."""
        return (
            np.unique(np.array([s.image.shape for s in self.scenes]), axis=0).shape[0]
            == 1
        )

    @property
    def consistent(self):
        """Specifies whether all scenes have the same annotation
        configuration."""
        return all(s.annotation_config == self.annotation_config for s in self.scenes)

    @property
    def images(self):
        """All the images for a scene collection.
        All images will be loaded if not already cached."""
        return [s.image for s in self.scenes]

    def augment(self, **kwargs):
        """Obtained an augmented version of the given collection.
        All arguments passed to `Scene.augment`"""
        return self.assign(scenes=[s.augment(**kwargs) for s in self.scenes])

    def to_tfrecords(self, output_prefix, n_scenes_per_shard=1):
        """Write scene collection as a series of TfRecord files.

        Args:
            output_prefix: The prefix for the .tfrecord files (e.g.,
                my_directory/my_training_dataset).
            n_scenes_per_shard: The number of scenes to store in each file.
        """
        if os.path.dirname(output_prefix):
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        n_shards = math.ceil(len(self) / n_scenes_per_shard)
        for shard, start in enumerate(range(0, len(self), n_scenes_per_shard)):
            with tf.io.TFRecordWriter(
                f"{output_prefix}-{shard + 1}-of-{n_shards}.tfrecord"
            ) as writer:
                for scene in [
                    self[idx]
                    for idx in range(start, min(start + n_scenes_per_shard, len(self)))
                ]:
                    writer.write(scene.to_example().SerializeToString())

    @classmethod
    def from_tfrecord_pattern(cls, pattern, annotation_config):
        """Load a scene collection from TfRecord files.

        Args:
            pattern: The file pattern for the TfRecord files (e.g.,
                my_directory/my_training_dataset*.tfrecord)
            annotation_config: The annotation configuration to use
                when loading the examples.
        """
        return cls(
            scenes=[
                Scene.from_example(record, annotation_config=annotation_config)
                for record in (
                    tf.data.Dataset.list_files(pattern).interleave(
                        tf.data.TFRecordDataset
                    )
                )
            ],
            annotation_config=annotation_config,
        )

    def train_test_split(
        self, *args, **kwargs
    ) -> typing.Tuple["SceneCollection", "SceneCollection"]:
        """Obtain new scene collections, split into train
        and test. All arguments passed to
        `sklearn.model_selection.train_test_split
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_.

        For example, to get two collections, one
        containing 70% of the data and the other containing
        30%, ::

            training, testing = collection.train_test_split(
                train_size=0.7, test_size=0.3
            )

        You can also use the `stratify` argument to ensure an even split
        between different kinds of scenes. For example, to split
        scenes containing at least 3 annotations proportionally, ::

            training, testing = collection.train_test_split(
                train_size=0.7, test_size=0.3,
                stratify=[len(s.annotations) >= 3 for s in collection]
            )


        Returns:
            A train and test scene collection.
        """
        train, test = train_test_split(self.scenes, *args, **kwargs)
        return (self.assign(scenes=train), self.assign(scenes=test))

    def assign(self, **kwargs) -> "SceneCollection":
        """Obtain a new scene with the given keyword arguments
        changing. If `annotation_config` is provided, the annotations
        are converted to the new `annotation_config` first.

        Returns:
            A new scene

        """
        if "annotation_config" in kwargs:
            annotation_config = kwargs["annotation_config"]
            scenes = kwargs.get("scenes", self.scenes)
            kwargs["scenes"] = [
                s.assign(annotation_config=annotation_config) for s in scenes
            ]
        defaults = {"scenes": self.scenes, "annotation_config": self.annotation_config}
        kwargs = {**defaults, **kwargs}
        return SceneCollection(**kwargs)

    def sample(self, n, replace=True) -> "SceneCollection":
        """Get a random subsample of this collection"""
        selected = np.random.choice(len(self.scenes), n, replace=replace)
        return self.assign(scenes=[self.scenes[i] for i in selected])

    def fit(self, **kwargs):
        """Obtain a new scene collection, fitted to the given width
        and height. All arguments passed to `scene.fit()`

        Returns:
            The new scene collection and list of scales.
        """
        scales = []
        scenes = []
        for s in self.scenes:
            scene, scale = s.fit(**kwargs)
            scenes.append(scene)
            scales.append(scale)
        return self.assign(scenes=scenes), scales

    def thumbnails(self, n=10, width=200, height=200, ncols=2):
        """Get a thumbnail sample of the images in the collection.

        Args:
            n: The number of images to include
            width: The width of each thumbnail
            height: The height of each thumbnail
            ncols: The number of columsn in which to arrange the
                thumbnails.

        Returns:
            The thumbnail image with width ncols*width and height
            (n / ncols)*height
        """
        if n > len(self.scenes):
            log.warning(
                "Collection only has %s scenes but you requested %s thumbnails.",
                len(self.scenes),
                n,
            )
            n = len(self.scenes)
        nrows = math.ceil(n / ncols)
        sample_indices = np.random.choice(len(self.scenes), n, replace=False)
        sample = [self.scenes[i] for i in sample_indices]
        thumbnails = [scene.annotated() for scene in sample]
        thumbnails = [t.fit(width=width, height=height)[0] for t in thumbnails]
        thumbnail = utils.get_blank_image(
            width=ncols * width, height=nrows * height, n_channels=3
        )
        for rowIdx in range(nrows):
            for colIdx in range(ncols):
                if len(thumbnails) == 0:
                    break
                thumbnail[
                    rowIdx * width : (rowIdx + 1) * width,
                    colIdx * height : (colIdx + 1) * height,
                ] = thumbnails.pop()
        return thumbnail
