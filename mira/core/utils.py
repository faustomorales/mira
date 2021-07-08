# pylint: disable=unsupported-assignment-operation
import typing
import os
import logging
import urllib
import urllib.request
import io

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import validators
import cv2

log = logging.getLogger(__name__)


def _get_channels(image):
    """Get the number of channels in the image"""
    return 0 if len(image.shape) == 2 else image.shape[2]


def _rbswap(image):
    """Swap the red and blue channels for reading and writing
    images."""
    if _get_channels(image) != 3:
        log.info("Not swapping red and blue due to number of channels.")
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


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
    return _rbswap(image)


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
    return np.uint8(image)


def color(image, n_channels=3):
    """Convert to color image if it is not already."""
    if len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        return image.repeat(n_channels, axis=2)
    return image


def scaled(image, minimum: float = -1, maximum: float = 1):
    """Obtain a scaled version of the image with values between
    minimum and maximum.

    Args:
        minimum: The minimum value
        maximum: The maximum value

    Returns:
        An array of same shape as image but of dtype `np.float32` with
        values scaled appropriately.
    """
    assert maximum > minimum
    x = np.float32(image)
    x /= 255
    x *= maximum - minimum
    x += minimum
    return x


def fit(image, width: int, height, cval: int = 255) -> typing.Tuple[np.ndarray, float]:
    """Obtain a new image, fit to the specified size.

    Args:
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image

    Returns:
        The new image and the scaling that was applied.
    """
    if width == image.shape[1] and height == image.shape[0]:
        return image, 1
    scale = min(width / image.shape[1], height / image.shape[0])
    fitted = get_blank_image(
        width=width, height=height, n_channels=_get_channels(image), cval=cval
    )
    image = resize(image, scale=scale)
    fitted[: image.shape[0], : image.shape[1]] = image[:height, :width]
    return fitted, scale


def resize(image, scale: float, interpolation=cv2.INTER_NEAREST):
    """Obtain resized version of image with a given scale

    Args:
        scale: The scale by which to resize the image
        interpolation: The interpolation method to use

    Returns:
        The scaled image
    """
    width = int(np.ceil(scale * image.shape[1]))
    height = int(np.ceil(scale * image.shape[0]))
    resized = cv2.resize(image, dsize=(width, height), interpolation=interpolation)
    if len(resized.shape) == 2 and len(image.shape) == 3:
        # This was a grayscale image and we need it to be returned
        # as such.
        resized = resized[:, :, np.newaxis]
    return resized


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
    image = _rbswap(image)
    if hasattr(filepath_or_buffer, "write"):
        data = cv2.imencode(extension, image)[1].tobytes()
        filepath_or_buffer.write(data)  # type: ignore
    else:
        cv2.imwrite(filepath_or_buffer, img=image)


def show(image, ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
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
