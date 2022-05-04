# pylint: disable=unsupported-assignment-operation
import io
import os
import typing
import logging
import collections
import urllib
import urllib.request

import typing_extensions
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import validators
import cv2

log = logging.getLogger(__name__)

MaskRegion = typing_extensions.TypedDict(
    "MaskRegion", {"visible": bool, "contour": np.ndarray, "name": str}
)


def read(filepath_or_buffer: typing.Union[str, io.BytesIO, typing.BinaryIO]):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)  # type: ignore
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    elif isinstance(filepath_or_buffer, str) and validators.url(filepath_or_buffer):
        with urllib.request.urlopen(filepath_or_buffer) as data:
            return read(data)
    else:
        assert os.path.isfile(
            filepath_or_buffer  # type: ignore
        ), "Could not find image at path: " + str(filepath_or_buffer)
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image


def get_blank_image(width: int, height: int, n_channels: int, cval=255) -> np.ndarray:
    """Obtain a new blank image with given dimensions.

    Args:
        width: The width of the blank image
        height: The height of the blank image
        n_channels: The number of channels. If 0, the image
            will only have two dimensions (y and x).
        cval: The value to set all pixels to (does not apply to the
            alpha channel)

    Returns:
        The blank image
    """
    if n_channels == 0:
        image = np.zeros((height, width)) + 255
    else:
        image = np.zeros((height, width, n_channels)) + cval
    return image.astype("uint8")


def save(
    image,
    filepath_or_buffer: typing.Union[str, io.BytesIO, typing.BinaryIO],
    extension=".jpg",
):
    """Save the image

    Args:
        filepath_or_buffer: The file or buffer to
            which to save the image. If buffer,
            format must be provided
        esxtension: The extension for the format to use
            if writing to buffer
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    if hasattr(filepath_or_buffer, "write"):
        data = cv2.imencode(extension, image)[1].tobytes()
        filepath_or_buffer.write(data)  # type: ignore
    else:
        cv2.imwrite(filepath_or_buffer, img=image)


def imshow(image, ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Show an image

    Args:
        ax: Axis on which to show the image

    Returns:
        An axes object
    """
    if ax is None:
        ax = plt
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return ax.imshow(image)
    if len(image.shape) == 3 and image.shape[2] == 1:
        return ax.imshow(image[:, :, 0])
    if len(image.shape) == 2:
        return ax.imshow(image)
    raise ValueError("Incorrect dimensions for image data.")


# pylint: disable=unexpected-keyword-arg
def compute_iou(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for two sets of boxes. Consider the following
    example:
    .. code-block:: python
        compute_iou(
            boxesA=np.array([[0, 0, 1, 1]]),
            boxesB=np.array([[0, 0, 0.5, 0.5], [1, 1, 2, 2]])
        )
    This would yield :code:[[0.25, 0]] because the first box has
    overlap of (0.5 * 0.5) divided by a union area of
    ((0.5 * 0.5) + (1 * 1) - (0.5*0.5)).
    Args:
        boxesA: The first set of boxes, provided as x1, y1, x2, y2
            coordinates as an array with shape (NA, 4).
        boxesB: The second set of boxes provided in same form as boxesA.
    Returns:
        An IoU matrix of shape (NA, NB) where 0 means no overlap and
        1 means complete overlap.
    """

    # A joint matrix for all box pairs so we can vectorize. It has shape
    # (NA, NB, 2, 4).
    boxes = np.zeros((boxesA.shape[0], boxesB.shape[0], 2, 4))
    boxes[:, :, 0] = boxesA[:, np.newaxis, :]
    boxes[:, :, 1] = boxesB[np.newaxis, :, :]
    xA = boxes[..., 0].max(axis=-1)
    yA = boxes[..., 1].max(axis=-1)
    xB = boxes[..., 2].min(axis=-1)
    yB = boxes[..., 3].min(axis=-1)
    interArea = (xB - xA).clip(0) * (yB - yA).clip(0)

    boxAArea = (boxes[..., 0, 2] - boxes[..., 0, 0]) * (
        boxes[..., 0, 3] - boxes[..., 0, 1]
    )
    boxBArea = (boxes[..., 1, 2] - boxes[..., 1, 0]) * (
        boxes[..., 1, 3] - boxes[..., 1, 1]
    )
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def compute_coverage(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
    """Compute pairwise overlap of two sets of boxes. Useful for detecting
    redundant bounding boxes where there is no score to consider (as in
    non-max suppression). Consider the following example:

    .. code-block:: python

        compute_overlap(
            boxesA=np.array([[0, 0, 1, 1]]),
            boxesB=np.array([[0, 0, 0.5, 0.5], [1, 1, 2, 2]])
        )
    This would yield :code:[[0.25, 0]] because 0.25 of boxA1 is contained
    within boxB1 while none of boxA1 is contained within box B2. If
    both boxes are for the same class, it may make sense to keep only
    the box that fully encompasses the other (this is problem-specific).
    Note that this function is not symmetric (i.e.,
    compute_coverage(x, y) != compute_coverage(y, x)). In our example,
    if we had reversed the inputs, we would have gotten :code:[[1], [0]]
    because 100% of boxB1 (now A1) is contained in boxA1 (now B1) while
    none of boxB2 (now A2) is contained in boxA1 (now B1).

    Args:
        boxesA: The first set of boxes, provided as x1, y1, x2, y2
            coordinates as an array with shape (NA, 4).
        boxesB: The second set of boxes provided in same form as boxesA.
    Returns:
        An IoU matrix of shape (NA, NB, 2) where 0 means no overlap and
        1 means complete overlap.
    """
    # A joint matrix for all box pairs so we can vectorize. It has shape
    # (NA, NB, 2, 4).
    boxes = np.zeros((boxesA.shape[0], boxesB.shape[0], 2, 4))
    boxes[:, :, 0] = boxesA[:, np.newaxis, :]
    boxes[:, :, 1] = boxesB[np.newaxis, :, :]
    xA = boxes[..., 0].max(axis=-1)
    yA = boxes[..., 1].max(axis=-1)
    xB = boxes[..., 2].min(axis=-1)
    yB = boxes[..., 3].min(axis=-1)
    interArea = (xB - xA).clip(0) * (yB - yA).clip(0)

    boxAArea = (boxes[..., 0, 2] - boxes[..., 0, 0]) * (
        boxes[..., 0, 3] - boxes[..., 0, 1]
    )
    coverageA = interArea / boxAArea
    return coverageA


def split(
    items: typing.List[typing.Any],
    sizes: typing.List[float],
    random_state: int = 42,
    stratify: typing.Sequence[typing.Hashable] = None,
    group: typing.Sequence[typing.Hashable] = None,
) -> typing.Sequence[typing.Any]:
    """Split a list of items into groups of specific proportions.

    For example, to split an array into groups containing 70%, 15%, and 15% of the
    dataset, respectively, you can do something like the following:

    .. code-block:: python

        training, validation, test = split(
            np.arange(100),
            sizes=[0.7, 0.15, 0.15]
        )

    You can also use the `stratify` argument to ensure an even split
    between different kinds of scenes. For example, to split
    scenes containing at least 3 annotations proportionally,
    do something like the following.

    .. code-block:: python

        training, validation, test = split(
            np.arange(100),
            sizes=[0.7, 0.15, 0.15],
            stratify=np.random.choice(np.arange(3), size=100)
        )

    Finally, you can make sure certain scenes end up in the same
    split (e.g., if they're crops from the same base image) using
    the group argument.

    .. code-block:: python

        training, validation, test = split(
            np.arange(100),
            sizes=[0.7, 0.15, 0.15],
            stratify=np.random.choice(np.arange(3), size=100),
            group=np.random.choice(np.arange(20), size=100)
        )

    Returns:
        A train and test scene collection.
    """
    if group is None:
        group = list(range(len(items)))
    if stratify is None:
        stratify = [0] * len(items)
    assert sum(sizes) == 1.0, "The sizes must add up to 1.0."
    assert len(group) == len(items), "group must be the same length as the collection."
    assert len(stratify) == len(
        items
    ), "stratify must be the same length as the collection."
    rng = np.random.default_rng(seed=random_state)
    unique = collections.Counter(group)
    hashes = [
        hash(tuple(set(s for s, g in zip(stratify, group) if g == u))) for u in unique
    ]
    totals = collections.Counter(hashes)
    assert len(unique) >= len(
        sizes
    ), "Cannot group when number of unique groups is less than len(sizes)."
    splits: typing.List[typing.List[typing.Any]] = [[] for _ in range(len(sizes))]
    for ht, t in totals.items():
        for a, u in zip(
            rng.choice(len(sizes), size=t, p=sizes),
            [u for h, u in zip(hashes, unique) if h == ht],
        ):
            splits[a].extend(i for i, g in zip(items, group) if g == u)
    return splits


def flatten(t):
    """Standard utility function for flattening a nested list taken from
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists."""
    return [item for sublist in t for item in sublist]


def find_largest_unique_boxes(
    bboxes, threshold=1, method: typing_extensions.Literal["iou", "coverage"] = "iou"
):
    """Find the largest entries in a list of bounding boxes that are not duplicative with
    a larger box.

    Args:
        bboxes: An array of bounding boxes in x1y1x2y2 format.
        method: Whether to check overlap by "coverage" (i.e.,
            is X% of box A contained by some larger box B) or "iou"
            (intersection-over-union). IoU is, of course, more strict.
        threshold: The threshold for equality. Boxes are retained if there
            is no larger box with which the overlap is greater than or
            equal to this threshold.
    """

    assert method in ["iou", "coverage"]
    func = compute_iou if method == "iou" else compute_coverage
    if len(bboxes) <= 1:
        # You can't have duplicates if there's only one or None.
        return np.array([0] if len(bboxes) == 1 else [])

    # Sort by area because we're going to identify duplicates
    # in order of size.
    indexes = np.argsort(np.product(bboxes[:, 2:] - bboxes[:, :2], axis=1))

    # Keep only annotations that are not duplicative with a larger (i.e.,
    # later in our sorted list) annotation. The largest annotation is, of course,
    # always retained.
    return np.array(
        [
            bidx
            for bidx, (cidx, coverages) in zip(
                indexes, enumerate(func(bboxes[indexes], bboxes[indexes]))
            )
            if (cidx == len(bboxes) - 1 or coverages[cidx + 1 :].max() < threshold)
        ]
    )


def apply_mask(image, masks):
    """Given an image and a list of masking contours,
    apply masking in place."""
    if not masks:
        return
    hide = [m for m in masks if not m["visible"]]
    show = [m for m in masks if m["visible"]]
    if show:
        # Assume something is masked, unless it's shown.
        mask = np.ones(image.shape[:2], dtype="uint8")
        for m in show:
            cv2.drawContours(
                mask,
                contours=[m["contour"]],
                contourIdx=-1,
                color=0,
                thickness=-1,
            )
    else:
        # Assume something is unmasked, unless it's hidden.
        mask = np.zeros(image.shape[:2], dtype="uint8")
    for m in hide:
        cv2.drawContours(
            mask,
            contours=[m["contour"]],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )
    image[mask > 0] = 0


def transform_bboxes(
    bboxes: np.ndarray, M: np.ndarray, width: int = None, height: int = None, clip=True
):
    """Transform a set of axis-aligned bounding boxes.

    Args:
        bboxes: An array of shape (N, 4) where each row is (xmin, ymin, xmax, ymax)
        M: A transform matrix of shape 2x3.
        width: The width of the output image (for clipping output boxes)
        height: The height of the output image (for clipping output boxes)
        clip: Whether to apply clipping.
    """
    if len(bboxes) == 0:
        return bboxes
    x1, y1, x2, y2 = bboxes.T
    vertices = np.array([x1, y1, x2, y1, x2, y2, x1, y2]).T.reshape((-1, 4, 2))
    transformed_vertices = (
        cv2.transform(vertices, m=M)
        if M.shape == (2, 3)
        else cv2.perspectiveTransform(vertices.astype("float32"), m=M)
    )
    if clip:
        assert (
            width is not None and height is not None
        ), "If clipping, width and height must be provided."
        transformed_vertices = transformed_vertices.clip(0, [width, height])
    transformed_bboxes = np.concatenate(
        [
            transformed_vertices.min(axis=1).T,
            transformed_vertices.max(axis=1).T,
        ],
        axis=0,
    ).T
    return transformed_bboxes
