# pylint: disable=unsupported-assignment-operation
import io
import os
import typing
import logging
import collections

import requests
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
ContourList = typing.List[np.ndarray]
DeduplicationMethod = typing_extensions.Literal["iou", "coverage"]


def box2pts(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Convert bounding box coordinates to a contour array."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def read(
    filepath_or_buffer: typing.Union[str, io.BytesIO, typing.BinaryIO, np.ndarray]
):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)  # type: ignore
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    elif isinstance(filepath_or_buffer, str) and validators.url(filepath_or_buffer):
        return read(io.BytesIO(requests.get(filepath_or_buffer).content))
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


def image2bytes(image: np.ndarray, extension=".png"):
    """Convert an RGB or grayscale image to PNG bytes."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
    return cv2.imencode(extension, image)[1].tobytes()


def save(
    image,
    filepath_or_buffer: typing.Union[str, io.BytesIO, typing.BinaryIO],
    extension=".png",
):
    """Save the image

    Args:
        filepath_or_buffer: The file or buffer to
            which to save the image. If buffer,
            format must be provided
        esxtension: The extension for the format to use
            if writing to buffer
    """
    data = image2bytes(image, extension=extension)
    if hasattr(filepath_or_buffer, "write"):
        filepath_or_buffer.write(data)  # type: ignore
    else:
        with open(filepath_or_buffer, "wb") as f:  # type: ignore
            f.write(data)


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


def compute_contour_binary_masks(
    contour1: np.ndarray, contour2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Given two contours, build binary images showing the coverage of eash, scaling them to a maximum size of 100px."""
    points = np.concatenate([contour1, contour2], axis=0)
    offset = points.min(axis=0)
    points, contour1, contour2 = [v - offset for v in [points, contour1, contour2]]
    scale = min(100 / points.max(axis=0).min(), 1)
    if scale < 1:
        points, contour1, contour2 = [v * scale for v in [points, contour1, contour2]]
    w, h = points.max(axis=0).astype("int32")
    im1, im2 = [
        cv2.drawContours(
            np.zeros((h, w), dtype="uint8"),
            contours=(box[np.newaxis]).round().astype("int32"),
            color=255,
            thickness=-1,
            contourIdx=0,
        )
        > 0
        for box in [contour1, contour2]
    ]
    return im1, im2


def compute_coverage_for_contour_pair(contour1: np.ndarray, contour2: np.ndarray):
    """Compute how much of contour1 is contained within contour2."""
    im1, im2 = compute_contour_binary_masks(contour1, contour2)
    return (im1 & im2).sum() / im1.sum()


def compute_iou_for_contour_pair(contour1: np.ndarray, contour2: np.ndarray):
    """Compute IoU for a pair of contours.

    Args:
        contour1: The first contour.
        contour2: The second contour.
    """
    im1, im2 = compute_contour_binary_masks(contour1, contour2)
    return (im1 & im2).sum() / (im1 | im2).sum()


def compute_contour_iou(contoursA: ContourList, contoursB: ContourList):
    """Compute pairwise IoU for two sets of contours.

    Args:
        contoursA: The first set of contours as a list of point arrays.
        contoursB: The second set of boxes provided in same form as contoursA.
    Returns:
        An IoU matrix of shape (NA, NB) where 0 means no overlap and
        1 means complete overlap.
    """
    return np.array(
        [
            [compute_iou_for_contour_pair(contour1, contour2) for contour2 in contoursB]
            for contour1 in contoursA
        ]
    )


def compute_contour_coverage(contoursA: ContourList, contoursB: ContourList):
    """Compute pairwise overlap of two sets of contours."""
    arr = np.zeros((len(contoursA), len(contoursB)), dtype="float32")
    for idx1, contour1 in enumerate(contoursA):
        for idx2, contour2 in enumerate(contoursB):
            arr[idx1, idx2] = compute_coverage_for_contour_pair(contour1, contour2)
    return arr


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


def find_largest_unique_contours(contours, threshold=1, method="iou"):
    """Find the largest entries in a list of contours that are not duplicative with a larger contour.
    Same as find_largest_unique_boxes but with contours."""
    assert method in ["iou", "coverage"]
    func = compute_contour_iou if method == "iou" else compute_contour_coverage
    if len(contours) <= 1:
        # You can't have duplicates if there's only one or None.
        return np.array([0] if len(contours) == 1 else [])

    # Sort by area because we're going to identify duplicates
    # in order of size.
    indexes = sorted(
        range(len(contours)), key=lambda idx: cv2.contourArea(contours[idx])
    )
    sorted_contours = [contours[idx] for idx in indexes]
    # Keep only annotations that are not duplicative with a larger (i.e.,
    # later in our sorted list) annotation. The largest annotation is, of course,
    # always retained.
    return np.array(
        [
            bidx
            for bidx, (cidx, coverages) in zip(
                indexes, enumerate(func(sorted_contours, sorted_contours))
            )
            if (cidx == len(contours) - 1 or coverages[cidx + 1 :].max() < threshold)
        ]
    )


def find_largest_unique_boxes(bboxes, threshold=1, method: DeduplicationMethod = "iou"):
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


def transform_contour(contour: np.ndarray, M):
    """Apply a transform to a contour."""
    return cv2.transform(contour[:, np.newaxis], m=M)[:, 0, :2]
