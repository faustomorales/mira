import logging
import typing

import numpy as np

from .selection import Selection

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
        selection: Selection,
        category: AnnotationCategory,
        score: float = None,
        metadata: dict = None,
    ):
        if category is None:
            raise ValueError("A category object must be specified.")
        self.selection = selection
        self.category = category
        self.score = score
        self.metadata = metadata or {}

    def assign(self, **kwargs) -> "Annotation":
        """Get a new Annotation with only the supplied
        keyword arguments changed."""
        defaults = {
            "selection": self.selection,
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

    def resize(self, scale) -> "Annotation":
        """Get a new annotation with the given
        uniform scaling."""
        return self.assign(selection=self.selection.resize(scale=scale))

    def __eq__(self, other):
        self_bbox = self.selection.x1y1x2y2()
        other_bbox = other.selection.x1y1x2y2()
        return self_bbox == other_bbox and self.category == other.category


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
            [
                list(a.selection.x1y1x2y2()) + [self.index(a.category)]
                for a in annotations
            ],
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

    @property
    def types(self):
        """Get the list of types."""
        return self._types

    def index(self, category):
        """Get the index for a category."""
        return next(i for i, cat in enumerate(self) if cat == category)
