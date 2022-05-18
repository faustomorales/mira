"""Scene and SceneCollection objects"""

# pylint: disable=invalid-name,len-as-condition,unsupported-assignment-operation

import os
import io
import json
import typing
import logging
import tarfile
import tempfile

import tqdm
import pandas as pd
import albumentations as A
import typing_extensions as tx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

from .protos import scene_pb2 as mps
from .annotation import AnnotationConfiguration, Annotation
from . import utils, augmentations

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
        masks: A list of MaskRegion dictonaries which will determine
            which parts of images are shown and hidden.
    """

    def __init__(
        self,
        annotation_config: AnnotationConfiguration,
        annotations: typing.List[Annotation],
        image: typing.Union[np.ndarray, str],
        metadata: dict = None,
        cache: bool = False,
        masks: typing.List[utils.MaskRegion] = None,
    ):
        assert isinstance(
            image, (np.ndarray, str)
        ), "Image must be string or ndarray, not " + str(type(image))
        if masks is None:
            masks = []
        self.metadata = metadata
        self._image = image
        self._annotations = annotations
        self._annotation_config = annotation_config
        self.cache = cache
        self.masks = masks

    @property
    def image(self) -> np.ndarray:
        """The image that is being annotated"""
        # Check to see if we have an actual image
        # or just a string
        protect_image = False
        if isinstance(self._image, str):
            # Load the image
            log.debug("Reading from %s", self._image)
            image = utils.read(self._image)
        else:
            protect_image = True
            log.debug("Reading image from cache.")
            image = self._image
        # Check how to handle caching the image
        # for future reads
        if self.cache is True:
            log.debug("Caching image.")
            protect_image = True
            self._image = image
        elif self.cache is False:
            pass
        else:
            raise ValueError(f"Unsupported cache parameter: {self.cache}.")
        if self.masks:
            if protect_image:
                # We should not modify this image. Work on a copy.
                image = image.copy()
            utils.apply_mask(image, masks=self.masks)
        return image

    @property
    def annotation_config(self):
        """The annotation configuration"""
        return self._annotation_config

    @property
    def annotations(self) -> typing.List[Annotation]:
        """Get the list of annotations"""
        return self._annotations

    @classmethod
    def fromString(cls, string):
        """Deserialize scene from string."""
        deserialized = mps.Scene.FromString(string)
        annotation_config = AnnotationConfiguration(
            deserialized.annotation_config.categories
        )
        image = cv2.imdecode(
            np.frombuffer(deserialized.image, dtype="uint8"), cv2.IMREAD_COLOR
        )
        annotations = []
        for annotation in deserialized.annotations:
            common = {
                "category": annotation_config[annotation.category],
                "metadata": json.loads(annotation.metadata),
            }
            if annotation.is_rect:
                annotations.append(
                    Annotation(
                        x1=annotation.x1,
                        y1=annotation.y1,
                        x2=annotation.x2,
                        y2=annotation.y2,
                        **common,
                    )
                )
            else:
                annotations.append(
                    Annotation(
                        points=np.array([[pt.x, pt.y] for pt in annotation.points]),
                        **common,
                    )
                )
        return cls(
            image=image,
            metadata=json.loads(deserialized.metadata),
            annotations=annotations,
            annotation_config=annotation_config,
            masks=[
                {
                    "visible": m.visible,
                    "name": m.name,
                    "contour": np.array([[p.x, p.y] for p in m.contour]),
                }
                for m in deserialized.masks
            ],
        )

    def toString(self):
        """Serialize scene to string."""
        return mps.Scene(
            image=cv2.imencode(".png", self.image)[1].tobytes(),
            annotation_config=mps.AnnotationConfiguration(
                categories=[c.name for c in self.annotation_config]
            ),
            metadata=json.dumps(self.metadata or {}),
            masks=[
                mps.Mask(
                    visible=m["visible"],
                    name=m["name"],
                    contour=[mps.Point(x=x, y=y) for x, y in m["contour"]],
                )
                for m in (self.masks or [])
            ],
            annotations=[
                mps.Annotation(
                    category=self.annotation_config.index(ann.category),
                    x1=ann.x1,
                    y1=ann.y1,
                    x2=ann.x2,
                    y2=ann.y2,
                    points=[mps.Point(x=x, y=y) for x, y in ann.points],
                    metadata=json.dumps(ann.metadata or {}),
                    is_rect=ann.is_rect,
                )
                for ann in self.annotations
            ],
        ).SerializeToString()

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
            "masks": self.masks,
        }
        kwargs = {**defaults, **kwargs}
        return Scene(**kwargs)

    def show(self, annotation_kwargs=None, **kwargs) -> mpl.axes.Axes:
        """Show an annotated version of the image. All arguments
        passed to `mira.core.utils.imshow()`.
        """
        return utils.imshow(self.annotated(**(annotation_kwargs or {})), **kwargs)

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

    def show_annotations(self, **kwargs):
        """Show annotations as individual plots. All arguments
        passed to plt.subplots."""
        if len(self.annotations) == 0:
            return None
        fig, axs = plt.subplots(nrows=len(self.annotations), **kwargs)
        if len(self.annotations) == 1:
            axs = [axs]
        image = self.image
        for ann, ax in zip(self.annotations, axs):
            ax.imshow(ann.extract(image))
            ax.set_title(ann.category.name)
        return fig

    def drop_duplicates(
        self, threshold=1, method: tx.Literal["iou", "coverage"] = "iou"
    ):
        """Remove annotations of the same class where one annotation covers similar or equal area as another.

        Args:
            method: Whether to check overlap by "coverage" (i.e.,
                is X% of box A contained by some larger box B) or "iou"
                (intersection-over-union). IoU is, of course, more strict.
            threshold: The threshold for equality. Boxes are retained if there
                is no larger box with which the overlap is greater than or
                equal to this threshold.
        """
        annotations = []
        for current_category in self.annotation_config:
            current_annotations = [
                ann for ann in self.annotations if ann.category == current_category
            ]
            # Keep only annotations that are not duplicative with a larger nnotation.
            annotations.extend(
                [
                    current_annotations[idx]
                    for idx in utils.find_largest_unique_boxes(
                        bboxes=self.annotation_config.bboxes_from_group(
                            current_annotations
                        )[:, :4],
                        method=method,
                        threshold=threshold,
                    )
                ]
            )
        return self.assign(annotations=annotations)

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
        img_raw = self.image.copy()
        img = img_raw
        for ann in self.annotations:
            ann.draw(img, color=color, opaque=opaque)
        utils.imshow(img, ax=ax)
        if labels:
            for ann in self.annotations:
                x1, y1, _, _ = ann.x1y1x2y2()
                ax.annotate(
                    ann.category.name,
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

    def augment(
        self, augmenter: augmentations.AugmenterProtocol = None, min_visibility=None
    ) -> typing.Tuple["Scene", np.ndarray]:
        """Obtain an augmented version of the scene using the given augmenter.

        Returns:
            The augmented scene
        """
        if augmenter is None:
            return self, np.eye(3)
        base_image = self.image
        base_points = np.array(
            [
                [0, 0],
                [base_image.shape[1], 0],
                [base_image.shape[1], base_image.shape[0]],
                [0, base_image.shape[0]],
            ]
        )
        transformed = augmenter(
            image=base_image,
            bboxes=[ann.x1y1x2y2() for ann in self.annotations],
            bbox_indices=[
                annIdx if ann.is_rect else -1
                for annIdx, ann in enumerate(self.annotations)
            ],
            keypoints=base_points.tolist()
            + utils.flatten(
                [ann.points.tolist() for ann in self.annotations if not ann.is_rect]
            ),
            keypoint_indices=[(None, None)] * 4
            + utils.flatten(
                [
                    [(annIdx, keyIdx) for keyIdx in range(len(ann.points))]
                    for annIdx, ann in enumerate(self.annotations)
                    if not ann.is_rect
                ]
            ),
        )

        image = transformed["image"]

        annotations = [
            Annotation(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                category=self.annotations[annIdx].category,
                metadata=self.annotations[annIdx].metadata,
            )
            for (x1, y1, x2, y2), annIdx in zip(
                transformed["bboxes"], transformed["bbox_indices"]
            )
            if annIdx > -1
        ] + [
            Annotation(
                points=keypoints.sort_values("keyIdx")[["x", "y"]].values,
                category=self.annotations[annIdx].category,
                metadata=self.annotations[annIdx].metadata,
            )
            for annIdx, keypoints in pd.concat(
                [
                    pd.DataFrame(
                        transformed["keypoint_indices"][4:],
                        columns=["annIdx", "keyIdx"],
                    ),
                    pd.DataFrame(transformed["keypoints"][4:], columns=["x", "y"]),
                ],
                axis=1,
            ).groupby(["annIdx"])
        ]
        recropped = [
            ann.crop(width=image.shape[1], height=image.shape[0]) for ann in annotations
        ]
        if min_visibility is None:
            min_visibility = (
                augmenter.processors["bboxes"].params.min_visibility
                if isinstance(augmenter, A.Compose)
                else 0.0
            )
        annotations = utils.flatten(
            [
                anns
                for ann, anns in zip(annotations, recropped)
                if ann.area() > 0
                and (sum((a.area() for a in anns), 0) / ann.area()) >= min_visibility
            ]
        )
        annotations = [ann for ann in annotations if ann.area() > 0]
        return self.assign(
            image=image,
            annotations=annotations,
        ), cv2.getPerspectiveTransform(
            src=np.array(base_points, dtype="float32")[:4],
            dst=np.array(transformed["keypoints"][:4], dtype="float32"),
        )

    def to_subcrops(self, max_size: int) -> typing.List["Scene"]:
        """Split a scene into subcrops of some maximum size while trying
        to avoid splitting annotations.

        Args:
            max_size: The maximum size of a crop (it may be smaller at the
                edges of an image).
        """
        if max_size % 2 == 0:
            r1 = r2 = max_size // 2
        else:
            r1 = max_size // 2
            r2 = max_size - r1
        annotations = self.annotations
        if not annotations:
            raise NotImplementedError("This function does not support empty scenes.")
        assert all(
            max(a.x2 - a.x1, a.y2 - a.y1) < max_size for a in annotations
        ), "At least one annotation is too big."
        image = self.image
        ih, iw = image.shape[:2]
        subcrops = []
        captured = []
        for annotation in annotations:
            if annotation in captured:
                # We already captured this annotation.
                continue
            # Get the others, sorted by distance to the
            # current annotation.
            axc, ayc = ((annotation.x1 + annotation.x2) / 2), (
                (annotation.y1 + annotation.y2) / 2
            )
            others = sorted(
                [a for a in annotations if a is not annotation],
                key=lambda a: np.square(
                    [
                        axc - ((a.x1 + a.x2) / 2),  # pylint: disable=cell-var-from-loop
                        ayc - ((a.y1 + a.y2) / 2),  # pylint: disable=cell-var-from-loop
                    ]
                ).sum(),
            )
            solved = False
            for r in range(len(others), -1, -1):
                # Try to fit the annotation and the r-closest other
                # annotations into this crop.
                ann_inc = [annotation] + others[:r]
                ann_exc = [a for a in annotations if a not in ann_inc]
                box_inc = self.annotation_config.bboxes_from_group(ann_inc)[:, :4]
                box_exc = self.annotation_config.bboxes_from_group(ann_exc)[:, :4]
                xmin, ymin = box_inc[:, :2].min(axis=0)
                xmax, ymax = box_inc[:, 2:].max(axis=0)
                if max(xmax - xmin, ymax - ymin) > max_size:
                    # This subset covers too large of an area.
                    continue
                xc, yc = map(
                    lambda v: max(v, r1), [(xmax + xmin) / 2, (ymax + ymin) / 2]
                )
                xc, yc = min(xc, iw - r2), min(yc, ih - r2)
                x1, y1, x2, y2 = map(round, [xc - r1, yc - r1, xc + r2, yc + r2])
                coverages = utils.compute_coverage(
                    np.concatenate([box_inc, box_exc], axis=0),
                    np.array([[x1, y1, x2, y2]]),
                )
                if (coverages[: len(box_inc), 0] == 1).all() and (
                    coverages[len(box_inc) :, 0] == 0
                ).all():
                    captured.extend(ann_inc)
                    subcrops.append(
                        self.assign(
                            image=image[y1:y2, x1:x2],
                            annotations=[
                                a.assign(
                                    x1=a.x1 - x1,
                                    y1=a.y1 - y1,
                                    x2=a.x2 - x1,
                                    y2=a.y2 - y1,
                                )
                                for a in ann_inc
                            ],
                        )
                    )
                    solved = True
                    break
            if not solved:
                raise ValueError("Failed to find a suitable crop.")
        return subcrops


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
        if annotation_config is None:
            annotation_config = scenes[0].annotation_config
        for i, s in enumerate(scenes):
            if s.annotation_config != annotation_config:
                raise ValueError(
                    f"Scene {i+1} of {len(scenes)} has inconsistent configuration."
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
    def annotation_sizes(self):
        """An array of dimensions for the annotations in the collection."""
        return np.diff(
            np.array(
                utils.flatten(
                    [[a.x1y1x2y2() for a in g] for g in self.annotation_groups]
                )
            ).reshape((-1, 2, 2)),
            axis=1,
        )[:, 0, :]

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

    def augment(self, augmenter: augmentations.AugmenterProtocol, **kwargs):
        """Obtained an augmented version of the given collection.
        All arguments passed to `Scene.augment`"""
        scenes, transforms = zip(
            *[s.augment(augmenter=augmenter, **kwargs) for s in self.scenes]
        )
        return self.assign(scenes=scenes), np.stack(transforms)

    def split(
        self,
        sizes: typing.List[float],
        random_state: int = 42,
        stratify: typing.Sequence[typing.Hashable] = None,
        group: typing.Sequence[typing.Hashable] = None,
    ) -> typing.Sequence["SceneCollection"]:
        """Obtain new scene collections, split based on a
        given set of proportios.

        For example, to get three collections containing 70%, 15%, and 15% of the
        dataset, respectively, you can do something like the following:

        .. code-block:: python

            training, validation, test = collection.split(
                sizes=[0.7, 0.15, 0.15]
            )

        You can also use the `stratify` argument to ensure an even split
        between different kinds of scenes. For example, to split
        scenes containing at least 3 annotations proportionally,
        do something like the following.

        .. code-block:: python

            training, validation, test = collection.split(
                sizes=[0.7, 0.15, 0.15],
                stratify=[len(s.annotations) >= 3 for s in collection]
            )

        Finally, you can make sure certain scenes end up in the same
        split (e.g., if they're crops from the same base image) using
        the group argument.

        .. code-block:: python

            training, validation, test = collection.split(
                sizes=[0.7, 0.15, 0.15],
                stratify=[len(s.annotations) >= 3 for s in collection],
                group=[s.metadata["origin"] for s in collection]
            )

        Returns:
            A train and test scene collection.
        """
        return [
            self.assign(scenes=scenes)
            for scenes in utils.split(
                self.scenes,
                sizes=sizes,
                random_state=random_state,
                stratify=stratify,
                group=group,
            )
        ]

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

    def save(self, filename: str):
        """Save scene collection a tarball."""
        with tarfile.open(filename, mode="w") as tar:
            for idx, scene in enumerate(tqdm.tqdm(self.scenes)):
                with tempfile.NamedTemporaryFile() as temp:
                    temp.write(scene.toString())
                    temp.flush()
                    tar.add(name=temp.name, arcname=str(idx))

    @classmethod
    def load(cls, filename: str, directory: str = None):
        """Load scene collection from a tarball. If a directory
        is provided, images will be saved into that directory
        rather than retained in memory."""
        if directory:
            os.makedirs(directory, exist_ok=True)
        scenes = []
        with tarfile.open(filename, mode="r") as tar:
            for idx, member in enumerate(tqdm.tqdm(tar.getmembers())):
                data = tar.extractfile(member)
                if data is None:
                    raise ValueError("Failed to load data from a file in the tarball.")
                if not directory:
                    scene = Scene.fromString(data.read())
                else:
                    label_filepath = os.path.join(directory, str(idx))
                    image_filepath = label_filepath + ".png"
                    if os.path.isfile(label_filepath) and os.path.isfile(
                        image_filepath
                    ):
                        with open(label_filepath, "rb") as f:
                            scene = Scene.fromString(f.read()).assign(
                                image=image_filepath
                            )
                    else:
                        scene = Scene.fromString(data.read())
                        cv2.imwrite(image_filepath, scene.image)
                        with open(label_filepath, "wb") as f:
                            f.write(
                                scene.assign(
                                    image=np.ones((1, 1, 3), dtype="uint8")
                                ).toString()
                            )
                    scene = scene.assign(image=image_filepath)
                scenes.append(scene)
        if len(scenes) == 0:
            raise ValueError("No scenes found.")
        return cls(scenes=scenes, annotation_config=scenes[0].annotation_config)
