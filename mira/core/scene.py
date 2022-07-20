"""Scene and SceneCollection objects"""

# pylint: disable=invalid-name,too-many-instance-attributes,len-as-condition,unsupported-assignment-operation,import-outside-toplevel

import os
import io
import json
import typing
import logging
import tarfile
import tempfile

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

from .protos import scene_pb2 as mps
from .annotation import Categories, Annotation, Label
from . import utils, augmentations, imagemeta
from ..thirdparty.albumentations import albumentations as A

log = logging.getLogger(__name__)

Dimensions = typing.NamedTuple("Dimensions", [("width", int), ("height", int)])


class Scene:
    """A single annotated image.

    Args:
        categories: The configuration for annotations for the
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

    _image: typing.Union[str, np.ndarray]
    _dimensions: typing.Optional[Dimensions]

    def __init__(
        self,
        categories: typing.Union[typing.List[str], Categories],
        image: typing.Union[np.ndarray, str],
        annotations: typing.List[Annotation] = None,
        metadata: dict = None,
        cache: bool = False,
        masks: typing.List[utils.MaskRegion] = None,
        labels: typing.List[Label] = None,
    ):
        assert isinstance(
            image, (np.ndarray, str)
        ), "Image must be string or ndarray, not " + str(type(image))
        if masks is None:
            masks = []
        self._image = image
        self._dimensions = None
        self.metadata = metadata
        self.annotations = annotations or []
        self.categories = Categories.from_categories(categories)
        self.labels = labels or []
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
    def image_bytes(self) -> bytes:
        """Get the image as a PNG encoded to bytes."""
        return utils.image2bytes(self.image)

    @classmethod
    def from_qsl(
        cls,
        item: typing.Dict,
        label_key: str,
        categories: Categories,
    ):
        """Create a scene from a set of QSL labels.

        Args:
            item: The QSL labeling item.
            label_key: The key for the region label to use
                for annotation.
            categories: The annotation configuration for the
                resulting scene.
        """
        import qsl

        target = item["target"]
        labels = item["labels"]
        dimensions = labels["dimensions"]
        annotations = []
        for box in labels["boxes"]:
            if not box["labels"].get(label_key):
                log.warning("A box in %s is missing %s. Skipping.", target, label_key)
                continue
            annotations.append(
                Annotation(
                    category=categories[box["labels"][label_key][0]],
                    x1=box["pt1"]["x"] * dimensions["width"],
                    y1=box["pt1"]["y"] * dimensions["height"],
                    x2=box["pt2"]["x"] * dimensions["width"],
                    y2=box["pt2"]["y"] * dimensions["height"],
                )
            )
        for mask in labels["masks"]:
            if not mask["labels"].get(label_key):
                log.warning("A mask in %s is missing %s. Skipping.", target, label_key)
                continue
            bitmap = qsl.counts2bitmap(**mask["map"])
            scaley, scalex = (
                dimensions["height"] / bitmap.shape[0],
                dimensions["width"] / bitmap.shape[1],
            )
            contours = cv2.findContours(
                bitmap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )[0]
            annotations.extend(
                [
                    Annotation(
                        category=categories[mask["labels"][label_key][0]],
                        points=contour[:, 0, :] * [scalex, scaley],
                    )
                    for contour in contours
                ]
            )
        for polygon in labels["polygons"]:
            if not polygon["labels"].get(label_key):
                log.warning(
                    "A polygon in %s is missing %s. Skipping.", target, label_key
                )
                continue
            annotations.append(
                Annotation(
                    category=categories[polygon["labels"][label_key][0]],
                    points=np.array([[p["x"], p["y"]] for p in polygon["points"]])
                    * [dimensions["width"], dimensions["height"]],
                )
            )
        return cls(
            image=target,
            annotations=annotations,
            categories=categories,
            metadata=item.get("metadata", {}),
        )

    @classmethod
    def fromString(cls, string):
        """Deserialize scene from string."""
        deserialized = mps.Scene.FromString(string)
        categories = Categories(deserialized.categories.categories)
        image = cv2.imdecode(
            np.frombuffer(deserialized.image, dtype="uint8"), cv2.IMREAD_COLOR
        )
        annotations = []
        for annotation in deserialized.annotations:
            common = {
                "category": categories[annotation.category],
                "metadata": json.loads(annotation.metadata),
                "score": annotation.score,
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
            labels=[
                Label(
                    category=categories[ann.category],
                    metadata=json.loads(ann.metadata),
                    score=ann.score,
                )
                for ann in deserialized.labels
            ],
            annotations=annotations,
            categories=categories,
            masks=[
                {
                    "visible": m.visible,
                    "name": m.name,
                    "contour": np.array([[p.x, p.y] for p in m.contour]),
                }
                for m in deserialized.masks
            ],
        )

    @property
    def dimensions(self) -> Dimensions:
        """Get size of image, attempting to get it without reading the entire file, if possible."""
        if self._dimensions is None:
            dimensions: typing.Optional[Dimensions] = None
            if isinstance(self._image, str):
                try:
                    log.info("Attempting to get dimensions from %s.", self._image)
                    meta = imagemeta.get_image_metadata(self._image)
                    dimensions = Dimensions(width=meta.width, height=meta.height)
                except Exception:  # pylint: disable=broad-except
                    log.info(
                        "Failed to load image metadata from disk for %s",
                        self._image,
                        exc_info=True,
                    )
            if dimensions is None:
                log.info("Loading dimensions from actual image.")
                image = self.image
                dimensions = Dimensions(width=image.shape[1], height=image.shape[0])
            self._dimensions = dimensions
        return self._dimensions

    def toString(self):
        """Serialize scene to string."""
        return mps.Scene(
            image=cv2.imencode(".png", self.image)[1].tobytes(),
            categories=mps.Categories(categories=[c.name for c in self.categories]),
            metadata=json.dumps(self.metadata or {}),
            masks=[
                mps.Mask(
                    visible=m["visible"],
                    name=m["name"],
                    contour=[mps.Point(x=x, y=y) for x, y in m["contour"]],
                )
                for m in (self.masks or [])
            ],
            labels=[
                mps.Label(
                    category=self.categories.index(ann.category),
                    score=ann.score,
                    metadata=json.dumps(ann.metadata or {}),
                )
                for ann in self.labels
            ],
            annotations=[
                mps.Annotation(
                    category=self.categories.index(ann.category),
                    x1=ann.x1,
                    y1=ann.y1,
                    x2=ann.x2,
                    y2=ann.y2,
                    score=ann.score,
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
        if "categories" in kwargs:
            # We need to change all the categories for annotations
            # to match the new annotation configuration.
            categories = kwargs["categories"]
            kwargs["annotations"] = [
                ann.convert(categories)
                for ann in kwargs.get("annotations", self.annotations)
            ]
            kwargs["labels"] = [
                ann.convert(categories) for ann in kwargs.get("labels", self.labels)
            ]
        # We use the _image instead of image to avoid triggering an
        # unnecessary read of the actual image.
        defaults = {
            "categories": self.categories,
            "annotations": self.annotations,
            "image": self._image,
            "cache": self.cache,
            "metadata": self.metadata,
            "masks": self.masks,
            "labels": self.labels,
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
        return self.categories.bboxes_from_group(self.annotations)

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
        fig.tight_layout()
        return fig

    def drop_duplicates(self, threshold=1, method: utils.DeduplicationMethod = "iou"):
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
        for current_category in self.categories:
            current_annotations = [
                ann for ann in self.annotations if ann.category == current_category
            ]
            # Keep only annotations that are not duplicative with a larger nnotation.
            annotations.extend(
                [
                    current_annotations[idx]
                    for idx in (
                        utils.find_largest_unique_boxes(
                            bboxes=self.categories.bboxes_from_group(
                                current_annotations
                            )[:, :4],
                            method=method,
                            threshold=threshold,
                        )
                        if all(ann.is_rect for ann in current_annotations)
                        else utils.find_largest_unique_contours(
                            contours=[ann.points for ann in current_annotations],
                            method=method,
                            threshold=threshold,
                        )
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
        # We include bbox placeholders for polygon annotations in order to
        # support mira.core.augmentations.RandomCropBboxSafe for now.
        transformed = augmenter(
            image=base_image,
            bboxes=[ann.x1y1x2y2() for ann in self.annotations],
            bbox_indices=[
                annIdx if ann.is_rect else -annIdx
                for annIdx, ann in enumerate(self.annotations, start=1)
            ],
            keypoints=base_points.tolist()
            + utils.flatten(
                [ann.points.tolist() for ann in self.annotations if not ann.is_rect]
            ),
            keypoint_indices=[(None, None)] * 4
            + utils.flatten(
                [
                    [(annIdx, keyIdx) for keyIdx in range(len(ann.points))]
                    for annIdx, ann in enumerate(self.annotations, start=1)
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
                category=self.annotations[annIdx - 1].category,
                metadata=self.annotations[annIdx - 1].metadata,
            )
            for (x1, y1, x2, y2), annIdx in zip(
                transformed["bboxes"], transformed["bbox_indices"]
            )
            if annIdx > 0
        ] + [
            Annotation(
                points=keypoints.sort_values("keyIdx")[["x", "y"]].values,
                category=self.annotations[annIdx - 1].category,
                metadata=self.annotations[annIdx - 1].metadata,
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
        transform = cv2.getPerspectiveTransform(
            src=np.array(base_points, dtype="float32")[:4],
            dst=np.array(transformed["keypoints"][:4], dtype="float32"),
        )
        augmented = self.assign(image=image, annotations=annotations, masks=[])
        return augmented, transform

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
                box_inc = self.categories.bboxes_from_group(ann_inc)[:, :4]
                box_exc = self.categories.bboxes_from_group(ann_exc)[:, :4]
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

    def compute_iou(self, other: "Scene"):
        """Obtain the inter-scene annotation IoU.

        Args:
            other: The other scene with which to compare.

        Returns:
            A matrix of shape (N, M) where N is the number of annotations
            in this scene and M is the number of annotations in the other
            scene. Each value represents the IoU between the two annotations.
            A negative IoU value means the annotations overlapped but they
            were for different classes.
        """
        iou = np.zeros((len(self.annotations), len(other.annotations)), dtype="float32")
        for idx1, ann1 in enumerate(self.annotations):
            for idx2, ann2 in enumerate(other.annotations):
                iou[idx1, idx2] = (
                    utils.compute_iou(
                        np.array([ann1.x1y1x2y2()]), np.array([ann2.x1y1x2y2()])
                    )[0, 0]
                    if ann1.is_rect and ann2.is_rect
                    else utils.compute_iou_for_contour_pair(ann1.points, ann2.points)
                ) * (1 if ann1.category.name == ann2.category.name else -1)
        return iou


class SceneCollection:
    """A collection of scenes.

    Args:
        categories: The configuration that should be used for all
            underlying scenes.
        scenes: The list of scenes.
    """

    def __init__(
        self,
        scenes: typing.List[Scene],
        categories: Categories = None,
    ):
        if categories is None:
            categories = scenes[0].categories
        for i, s in enumerate(scenes):
            if s.categories != categories:
                raise ValueError(
                    f"Scene {i+1} of {len(scenes)} has inconsistent configuration."
                )
        self._categories = Categories.from_categories(categories)
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
    def categories(self):
        """The annotation configuration"""
        return self._categories

    @property
    def annotation_groups(self):
        """The groups of annotations in the collection."""
        return [s.annotations for s in self.scenes]

    @property
    def label_groups(self):
        """The groups of labels in the collection."""
        return [s.labels for s in self.scenes]

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
        return all(s.categories == self.categories for s in self.scenes)

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
        changing. If `categories` is provided, the annotations
        are converted to the new `categories` first.

        Returns:
            A new scene

        """
        if "categories" in kwargs:
            categories = kwargs["categories"]
            scenes = kwargs.get("scenes", self.scenes)
            kwargs["scenes"] = [s.assign(categories=categories) for s in scenes]
        defaults = {"scenes": self.scenes, "categories": self.categories}
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

    def save_placeholder(
        self, filename: str, colormap: typing.Dict[str, typing.Tuple[int, int, int]]
    ):
        """Create a placeholder scene collection representing
        blank images with black blobs drawn on in the location of
        annotations. Useful for testing whether a detector has
        any chance of working with a given dataset.

        Args:
            filename: The tarball to which the dummy dataast should be
                saved.
            colormap: A mapping of annotation categories to colors, used
                for drawing the annotations onto a canvas.
        """
        with tarfile.open(filename, mode="w") as tar:
            for idx, scene in enumerate(tqdm.tqdm(self.scenes)):
                with tempfile.NamedTemporaryFile() as temp:
                    image = np.zeros_like(scene.image) + 255
                    for ann in typing.cast(Scene, scene).annotations:
                        ann.draw(image, color=colormap[ann.category.name], opaque=True)
                    temp.write(
                        scene.assign(
                            image=image, annotations=scene.annotations
                        ).toString()
                    )
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
                        cv2.imwrite(
                            image_filepath, cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR)
                        )
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
        return cls(scenes=scenes, categories=scenes[0].categories)

    @classmethod
    def from_qsl(cls, jsonpath: str, label_key: str):
        """Build a scene collection from a QSL JSON project file."""
        with open(jsonpath, "r", encoding="utf8") as f:
            project = json.loads(f.read())
        rconfig = next(
            (r for r in project["config"]["regions"] if r["name"] == label_key), None
        )
        if rconfig is None:
            raise ValueError(f"{label_key} region configuration not found.")
        categories = Categories([o["name"] for o in rconfig["options"]])
        scenes = []
        for item in project["items"]:
            if item.get("type", "image") != "image":
                log.info("Skipping item with type %s.", item["type"])
                continue
            if item.get("ignore", False):
                log.info("Skipping %s because it was ignored.", item["target"])
                continue
            if "labels" not in item:
                log.info("Skipping %s because labels are missing.", item["target"])
                continue
            scenes.append(
                Scene.from_qsl(item=item, label_key=label_key, categories=categories)
            )
        return cls(scenes=scenes)
