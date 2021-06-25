import typing

import cv2
import numpy as np


class Selection:
    """Defines a selection within an image.

    Args:
        points: A list of points defining the selection.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.y1, self.x2, self.y2 = map(int, [x1, y1, x2, y2])

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
        return Selection(
            *[
                max(0, min(v, d))
                for v, d in [
                    (self.x1, width),
                    (self.y1, height),
                    (self.x2, width),
                    (self.y2, height),
                ]
            ]
        )

    def resize(self, scale: float) -> "Selection":
        """Obtain a revised selection with a given
        uniform scaling."""
        return Selection(
            *[int(v * scale) for v in [self.x1, self.y1, self.x2, self.y2]]
        )
