from typing import Union, List, Tuple
from os import path
import numpy as np
import logging
import math
import io

from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib as mpl
import imgaug as ia
import validators

from .annotation import AnnotationConfiguration, Annotation
from .image import Image

log = logging.getLogger(__name__)


class Scene:
    """A single annotated image.

    Args:
        annotation_config: The configuration for annotations for the
            image.
        annotations: The list of annotations.
        image: The image that was annotated. Can be lazy-loaded by passing
            a string filepath.
        cache: Defines caching behavior for the image. If `True`, image is
            loaded into memory the first time that the image is requested.
            If `False`, image is loaded from the file path or URL whenever
            the image is requested.
    """

    def __init__(self,
                 annotation_config: AnnotationConfiguration,
                 annotations: List[Annotation],
                 image: Union[Image, np.ndarray, str],
                 cache: bool = True):
        assert isinstance(image, np.ndarray) or isinstance(image, str), \
            'Image must be string or ndarray, not ' + str(type(image))
        if isinstance(image, np.ndarray):
            image = image.view(Image)
        self._image = image
        self._annotations = annotations
        self._annotation_config = annotation_config
        self.cache = cache

    @property
    def image(self) -> Image:
        """The image that is being annotated"""
        # Check to see if we have an actual image
        # or just a string
        if type(self._image) != str:
            return self._image

        # Check the cache first if image is a URL
        if (validators.url(self._image) and type(self.cache) == str
                and path.isfile(self.cache)):
            self._image = self.cache

        # Load the image
        log.debug('Reading from ' + self._image)
        image = Image.read(self._image)

        # Check how to handle caching the image
        # for future reads
        if self.cache is True:
            self._image = image
        elif self.cache is False:
            pass
        else:
            raise ValueError('Cannot handle cache parameter: {0}.'.format(
                self.cache))
        return image

    @property
    def annotation_config(self):
        """The annotation configuration"""
        return self._annotation_config

    @property
    def annotations(self) -> List[Annotation]:
        """Get the list of annotations"""
        return self._annotations

    def assign(self, **kwargs) -> 'Scene':
        """Get a new scene with only the supplied
        keyword arguments changed."""
        if 'annotation_config' in kwargs:
            # We need to change all the categories for annotations
            # to match the new annotation configuration.
            annotations = kwargs.get('annotations', self.annotations)
            annotation_config = kwargs['annotation_config']
            revised = [
                ann.convert(annotation_config=annotation_config)
                for ann in annotations
            ]
            revised = [ann for ann in revised if ann is not None]
            removed = len(annotations) - len(revised)
            log.debug('Removed {0} annotations when changing '.format(removed)
                      + 'annotation configuration.')
            kwargs['annotations'] = revised
        # We use the _image instead of image to avoid triggering an
        # unnecessary read of the actual image.
        defaults = {
            'annotation_config': self.annotation_config,
            'annotations': self.annotations,
            'image': self._image,
            'cache': self.cache
        }
        kwargs = {**defaults, **kwargs}
        return Scene(**kwargs)

    def show(self, *args, **kwargs) -> mpl.axes.Axes:
        """Show an annotated version of the image. All arguments
        passed to `Image.show()`.
        """
        return self.annotated().show(*args, **kwargs)

    def scores(self):
        """Obtain an array containing the confidence
        score for each annotation."""
        return np.array([a.score for a in self.annotations])

    def bboxes(self):
        """Obtain an array of shape (N, 5) where the columns are
        x1, y1, x2, y2, class_index where class_index is determined
        from the annotation configuration."""
        return np.array([
            a.selection.bbox() + [self.annotation_config.index(a.category)
                                  ]  # noqa: E501
            for a in self.annotations
        ])

    def fit(self, width, height):
        """Obtain a new scene fitted to the given width and height.

        Args:
            width: The new width
            height: The new height

        Returns:
            The new scene and the scale
        """
        image, scale = self.image.fit(width=width, height=height)
        annotations = [ann.resize(scale=scale) for ann in self.annotations]
        return self.assign(image=image, annotations=annotations), scale

    def annotated(self,
                  dpi=72,
                  fontsize='x-large',
                  labels=True,
                  opaque=False,
                  color=(255, 0, 0)) -> Image:
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
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.axis('off')
        img_raw = self.image
        img = img_raw
        for ann in self.annotations:
            img = ann.selection.draw(img, color=color, opaque=opaque)
        img.show(ax=ax)
        if labels:
            for ann in self.annotations:
                x1, y1, x2, y2 = ann.selection.bbox()
                ax.annotate(
                    s=ann.category.name,
                    xy=(x1, y1),
                    fontsize=fontsize,
                    backgroundcolor=(1, 1, 1, 0.5))
        ax.set_xlim(0, img_raw.width)
        ax.set_ylim(img_raw.height, 0)
        fig.canvas.draw()
        raw = io.BytesIO()
        fig.savefig(
            raw,
            dpi=dpi,
            frameon=True,
            pad_inches=0,
            transparent=False,
            bbox_inches='tight')
        plt.close(fig)
        plt.ion()
        raw.seek(0)
        img = Image.read(raw)
        img = img[:, :, :3]
        raw.close()
        return img

    def resize(self, scale) -> 'Scene':
        """Obtain a resized version of the scene.

        Args:
            scale: The scale for the new scene.
        """
        return self.assign(
            image=self.image.resize(scale),
            annotations=[ann.resize(scale) for ann in self.annotations])

    def augment(self, augmenter: iaa.Augmenter = None,
                threshold: float = 0.25) -> 'Scene':
        """Obtain an augmented version of the scene using the given augmenter.

        Returns:
            The augmented scene
        """
        if augmenter is None:
            return self
        aug = augmenter.to_deterministic()
        keypoints = []
        keypoints_map = {}
        for i, ann in enumerate(self.annotations):
            current = ann.selection.keypoints()
            startIdx = len(keypoints)
            endIdx = startIdx + len(current)
            keypoints_map[i] = (startIdx, endIdx)
            keypoints.extend(current)
        keypoints = ia.KeypointsOnImage(keypoints, shape=self.image.shape)
        image = aug.augment_images([self.image])[0].view(Image)
        keypoints = aug.augment_keypoints([keypoints])[0].keypoints
        annotations = []
        for i, ann in enumerate(self.annotations):
            startIdx, endIdx = keypoints_map[i]
            current = keypoints[startIdx:endIdx]
            selection = ann.selection.assign_keypoints(current)
            area = selection.area()
            selection = selection.crop(width=image.width, height=image.height)
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

    def __init__(self,
                 scenes: List[Scene],
                 annotation_config: AnnotationConfiguration = None):
        assert len(scenes) > 0, \
            'A scene collection must have at least one scene'
        if annotation_config is None:
            annotation_config = scenes[0].annotation_config
        for i, s in enumerate(scenes):
            if s.annotation_config != annotation_config:
                raise ValueError(
                    'Scene {0} of {1} has inconsistent configuration.'.format(
                        i + 1, len(scenes)))
        self._annotation_config = annotation_config
        self._scenes = scenes

    def __getitem__(self, key):
        return self.scenes[key]

    def __len__(self):
        return len(self._scenes)

    @property
    def scenes(self):
        """The list of scenes"""
        return self._scenes

    @property
    def annotation_config(self):
        """The annotation configuration"""
        return self._annotation_config

    @property
    def uniform(self):
        """Specifies whether all scenes in the collection are
        of the same size. Note: This will trigger an image load."""
        return np.unique(
            np.array([s.image.shape for s in self.scenes]),
            axis=0).shape[0] == 1  # noqa: E501

    @property
    def consistent(self):
        """Specifies whether all scenes have the same annotation
        configuration."""
        return all(s.annotation_config == self.annotation_config
                   for s in self.scenes)  # noqa: E501

    @property
    def images(self):
        return [s.image for s in self.scenes]

    def augment(self, **kwargs):
        """Obtained an augmented version of the given collection.
        All arguments passed to `Scene.augment`"""
        return self.assign(scenes=[s.augment(**kwargs) for s in self.scenes])

    def train_test_split(self, *args, **kwargs
                         ) -> Tuple['SceneCollection', 'SceneCollection']:
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

    def assign(self, **kwargs) -> 'Scene':
        """Obtain a new scene with the given keyword arguments
        changing. If `annotation_config` is provided, the annotations
        are converted to the new `annotation_config` first.

        Returns:
            A new scene

        """
        if 'annotation_config' in kwargs:
            annotation_config = kwargs['annotation_config']
            scenes = kwargs.get('scenes', self.scenes)
            kwargs['scenes'] = [
                s.assign(annotation_config=annotation_config) for s in scenes
            ]
        defaults = {
            'scenes': self.scenes,
            'annotation_config': self.annotation_config
        }
        kwargs = {**defaults, **kwargs}
        return SceneCollection(**kwargs)

    def sample(self, n, replace=True) -> 'SceneCollection':
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
            log.warning('Collection only has {0} scenes but '
                        'you requested {1} thumbnails. Limiting '
                        'to {0} thumbnails.'.format(len(self.scenes), n))
            n = len(self.scenes)
        nrows = math.ceil(n / ncols)
        sample_indices = np.random.choice(len(self.scenes), n, replace=False)
        sample = [self.scenes[i] for i in sample_indices]
        thumbnails = [scene.annotated() for scene in sample]
        thumbnails = [t.fit(width=width, height=height)[0] for t in thumbnails]
        thumbnail = Image.new(
            width=ncols * width, height=nrows * height, channels=3)
        for rowIdx in range(nrows):
            for colIdx in range(ncols):
                if len(thumbnails) == 0:
                    break
                thumbnail[rowIdx * width:(rowIdx + 1) * width, colIdx *
                          height:(colIdx + 1) * height] = thumbnails.pop()
        return thumbnail
