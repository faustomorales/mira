import logging
import typing

import cv2
import numpy as np

log = logging.getLogger(__name__)


class AnnotationCategory:
    """Defines a category of an annotation along
    with all associated properties.

    Args:
        name: The name of the annotation category

    """

    def __init__(self, name: str):
        self._name = name

    def __eq__(self, other):
        return self._name == other._name

    @property
    def name(self):
        """The name of the category."""
        return self._name

    def __repr__(self):
        return repr(self._name)


class Annotation:
    """Defines a single annotation.

    Args:
        selection: The selection associated with the annotation
        category: The category of the annotation
        score: A score for the annotation
        metadata: Metadata to store as part of the annotation
    """

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        category: AnnotationCategory,
        score: float = None,
        metadata: dict = None,
    ):
        if category is None:
            raise ValueError("A category object must be specified.")
        self.x1, self.y1, self.x2, self.y2 = map(
            lambda v: int(round(v)), [x1, y1, x2, y2]
        )
        self.category = category
        self.score = score
        self.metadata = metadata or {}

    def area(self) -> int:
        """Compute the area of the selection."""
        return (self.y2 - self.y1) * (self.x2 - self.x1)

    def xywh(self) -> typing.Tuple[int, int, int, int]:
        """Get the bounding box as x, y, width
        and height."""
        x1, y1, x2, y2 = self.x1y1x2y2()
        return x1, y1, x2 - x1, y2 - y1

    def x1y1x2y2(self) -> typing.Tuple[int, int, int, int]:
        """The simple bounding box containing the selection.

        Returns:
            The coordinates (x1, y1, x2, y2). The first set always
            correspond with the top left of the image. The second
            set always correspond with the bottom right of the image.
        """
        return (self.x1, self.y1, self.x2, self.y2)

    def draw(
        self,
        image: np.ndarray,
        color: typing.Union[
            typing.Tuple[int, int, int], typing.Tuple[int, int, int, int]
        ],
        opaque: bool = False,
    ) -> np.ndarray:
        """Draw selection onto given image.

        Args:
            image: The image to draw on.
            color: The color to use.
            opaque: Whether the box should be filled.

        Returns:
            The image with the selection drawn
        """
        target = image.copy()
        pts = np.array(
            [
                [self.x1, self.y1],
                [self.x2, self.y1],
                [self.x2, self.y2],
                [self.x1, self.y2],
            ]
        )
        if opaque:
            return cv2.fillPoly(img=target, pts=[pts], color=color)
        return cv2.polylines(
            img=target, pts=[pts], isClosed=True, thickness=5, color=color
        )

    def extract(self, image):
        """Extract selection from image (i.e., crop the image
        to the selection).
        """
        x1, y1, x2, y2 = self.x1y1x2y2()
        cropped = image[max(y1, 0) : max(y2, 0), max(x1, 0) : max(0, x2)]
        return cropped

    def crop(self, width, height):
        """Crop a selection to a given image width
        and height.

        Args:
            width: The width of the image
            height: The height of the image
        """
        return self.assign(
            **{
                k: max(0, min(getattr(self, k), d))
                for d, k in [
                    (width, "x1"),
                    (height, "y1"),
                    (width, "x2"),
                    (height, "y2"),
                ]
            }
        )

    def resize(self, scale: float) -> "Annotation":
        """Obtain a revised selection with a given
        uniform scaling."""
        return self.assign(
            **{
                k: int(getattr(self, k) * scale)
                for k in [
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            }
        )

    def assign(self, **kwargs) -> "Annotation":
        """Get a new Annotation with only the supplied
        keyword arguments changed."""
        defaults = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "category": self.category,
            "score": self.score,
            "metadata": self.metadata,
        }
        kwargs = {**defaults, **kwargs}
        return Annotation(**kwargs)

    def convert(self, annotation_config) -> "Annotation":
        """Convert an annotation to match another annotation config."""
        name = self.category.name
        if name in annotation_config:
            return self.assign(
                category=annotation_config[name],
            )
        raise ValueError("%s is not in the new annotation configuration.")

    def __eq__(self, other):
        self_bbox = self.x1y1x2y2()
        other_bbox = other.x1y1x2y2()
        return self_bbox == other_bbox and self.category == other.category

    def __repr__(self):
        return repr(
            {
                "selection": {
                    "x1": self.x1,
                    "y1": self.y1,
                    "x2": self.x2,
                    "y2": self.y2,
                },
                "category": self.category.name,
                "score": self.score,
                "metadata": self.metadata,
            }
        )


class AnnotationConfiguration:
    """A class defining a list of annotation
    types for an object detection class.

    Args:
        names: The list of class names
    """

    def __init__(self, names: typing.List[str]):
        names = [s.lower() for s in names]
        if len(names) != len(set(names)):
            raise ValueError("All class names must be unique " "(case-insensitive).")
        self._types = [AnnotationCategory(name=name) for name in names]

    def bboxes_from_group(self, annotations: typing.List[Annotation]):
        """Obtain an array of shape (N, 5) where the columns are
        x1, y1, x2, y2, class_index where class_index is determined
        from the annotation configuration."""
        return np.array(
            [list(a.x1y1x2y2()) + [self.index(a.category)] for a in annotations],
        ).reshape(-1, 5)

    def __getitem__(self, key):
        if isinstance(key, np.int64):
            key = int(key)
        if isinstance(key, int):
            if key >= len(self):
                raise ValueError(
                    "Index {0} is out of bounds ".format(key)
                    + "(only have {0} entries)".format(len(self))
                )
            return self.types[key]
        if isinstance(key, str):
            key = key.lower()
            val = next((e for e in self._types if e.name == key), None)
            if val is None:
                raise ValueError("Did not find {0} in configuration".format(key))
            return val
        raise ValueError(f"Key must be int or str, not {key} of type {str(type(key))}")

    def __iter__(self):
        return iter(self._types)

    def __contains__(self, key):
        if isinstance(key, str):
            return any(e.name == key for e in self._types)
        if isinstance(key, AnnotationCategory):
            return any(e == key for e in self._types)
        raise ValueError("Key must be str or AnnotationCategory, not " + str(type(key)))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if len(other) != len(self):
            return False
        return all(o == s for s, o in zip(self, other))

    def __len__(self):
        return len(self._types)

    def __repr__(self):
        return repr([a.name for a in self.types])

    @property
    def types(self):
        """Get the list of types."""
        return self._types

    def index(self, category):
        """Get the index for a category."""
        return next(i for i, cat in enumerate(self) if cat == category)
