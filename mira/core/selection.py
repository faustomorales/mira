from typing import Union, Tuple, List
import matplotlib.path as mplPath

import cv2
import numpy as np
import imgaug as ia
from shapely import affinity, geometry
from scipy import spatial

from .image import Image


class Selection:
    """Defines a selection within an image.

    Args:
        points: A list of points defining the selection.
    """
    def __init__(self, points):
        assert len(points) > 1, 'A selection requires at least two points.'
        points = np.float32(points)
        if len(points) == 2:
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            points = np.float32([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])

        points = cv2.convexHull(points)[:, 0, :]
        self.points = points
        self.bbPath = mplPath.Path(points)

    def __add__(self, other):
        mp1 = geometry.MultiPoint(self.points)
        mp2 = geometry.MultiPoint(other.points)
        x, y = mp1.union(mp2).convex_hull.exterior.xy
        points = np.concatenate([
            np.expand_dims(x, axis=1),
            np.expand_dims(y, axis=1)
        ], axis=1)
        return Selection(points)

    def area(self) -> float:
        """Compute the area of the selection."""
        return cv2.contourArea(
            np.int32(self.points[:, np.newaxis, :])
        )

    def xywh(self) -> Tuple[int, int, int, int]:
        """Get the bounding box as x, y, width
        and height."""
        x1, y1, x2, y2 = self.bbox()
        return x1, y1, x2 - x1, y2 - y1

    def shrink(self, scale: float) -> 'Selection':
        """Obtain a new shrunk version of the selection.

        Args:
            scale: The amount by which to shrink the
                selection. Values less than 1 cause
                shrinking. Values larger than 1 cause
                expansion.
        """
        shrunk = affinity.scale(
            geometry.Polygon(self.points), xfact=scale, yfact=scale
        )
        return Selection(shrunk.exterior.coords)

    def contains_point(self, other) -> bool:
        """Determine whether this selection contains a
        particular point.

        Args:
            other: An iterable describing a point (x, y)

        Returns:
            A bool, True if `other` is contained within
            selection, False otherwise.
        """
        return self.bbPath.contains_point(other)

    def contains_points(self, other):
        """Determine whether this selection contains a
        series of points.

        Args:
            other: A numpy array of shape (N, 2)

        Returns:
            A numpy array of shape (N, ), describing
            whether or not the point at that
            position was in the selection.
        """
        return self.bbPath.contains_points(other)

    def bbox(self) -> Tuple[int, int, int, int]:
        """The simple bounding box containing the selection.

        Returns:
            The coordinates (x1, y1, x2, y2). The first set always
            correspond with the top left of the image. The second
            set always correspond with the bottom right of the image.
        """
        x1, y1 = self.points.min(axis=0)
        x2, y2 = self.points.max(axis=0)
        return [int(v) for v in [x1, y1, x2, y2]]

    def keypoints(self) -> List[ia.Keypoint]:
        """Obtain a list of keypoints for the selection.

        Returns:
            A list of keypoints
        """
        return [ia.Keypoint(x=x, y=y) for x, y in self.points]

    def draw(
        self,
        image: Image,
        color: Union[
            Tuple[int, int, int],
            Tuple[int, int, int, int]
        ],
        opaque: bool=False
    ) -> Image:
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
            [tuple(map(int, (x, y))) for x, y in self.points]
        )
        if opaque:
            return cv2.fillPoly(
                img=target,
                pts=[pts],
                color=color
            )
        else:
            return cv2.polylines(
                img=target,
                pts=[pts],
                isClosed=True,
                thickness=5,
                color=color
            )

    def extract(self, image):
        """Extract selection from image (i.e., crop the image
        to the selection).
        """
        x1, y1, x2, y2 = self.bbox()
        cropped = image[max(y1, 0):max(y2, 0), max(x1, 0):max(0, x2)]
        return cropped

    def assign_keypoints(self, keypoints: ia.KeypointsOnImage) -> 'Selection':
        """Obtain a revised version of the selection
        with the given keypoints"""
        if len(keypoints) != len(self.keypoints()):
            raise ValueError(
                'The wrong number of keypoints were provided.'
            )
        points = [
            (kp.x, kp.y) for kp in keypoints
        ]
        return Selection(
            points=points
        )

    def crop(self, width, height):
        """Crop a selection to a given image width
        and height.

        Args:
            width: The width of the image
            height: The height of the image
        """
        pts = self.points.copy()
        pts[:, 0] = pts[:, 0].clip(0, width)
        pts[:, 1] = pts[:, 1].clip(0, height)
        return Selection(points=pts)

    def resize(self, scale: float) -> 'Selection':
        """Obtain a revised selection with a given
        uniform scaling."""
        return Selection(
            points=[
                (scale*x, scale*y) for x, y in self.points
            ]
        )

    def rbox(self) -> Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            float
    ]:
        """Obtain the parameters of a rotated box.

        Returns:
            The vertices of the rotated box in top-left,
            top-right, bottom-right, bottom-left order along
            with the angle of rotation about the bottom left corner.
        """
        mp = geometry.MultiPoint(points=self.points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501

        # The code below is taken from
        # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        pts = np.array([tl, tr, br, bl], dtype="float32")

        rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
        return pts, rotation