import typing
import itertools

import cv2
import numpy as np
import scipy.sparse.csgraph as ssc

from . import utils


def collapse_boxes(boxes: np.ndarray, threshold=0.5, mode="smallest"):
    """Given a set of boxes, collapse overlapping boxes into the smallest
    or largest common area."""
    assert mode in ["smallest", "largest"], f"Unknown mode: {mode}"
    n_components, labels = ssc.connected_components(
        utils.compute_iou(boxes, boxes) > threshold, directed=False
    )
    return np.array(
        [
            (
                [subgroup[:, :2].max(axis=0), subgroup[:, 2:].min(axis=0)]
                if (
                    (subgroup[:, 2:].min(axis=0) - subgroup[:, :2].max(axis=0)) > 0
                ).all()
                else subgroup[
                    np.prod(subgroup[:, 2:] - subgroup[:, :2], axis=1).argmin()
                ]
            )
            if mode == "smallest"
            else [subgroup[:, :2].min(axis=0), subgroup[:, 2:].max(axis=0)]
            for subgroup in [
                boxes[labels == component] for component in range(n_components)
            ]
        ]
    ).reshape(-1, 4)


def find_consensus_regions(
    bbox_groups: typing.List[np.ndarray], iou_threshold: float = 0.5
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Given a set of box groups for the same image being
    labeled by different people, find the regions of
    consensus and non-consensus."""
    exclude_list = []
    for (bboxes1, bboxes2), annIdx in itertools.product(
        itertools.combinations(bbox_groups, 2),
        range(max(g[:, -1].max() if len(g) > 0 else 0 for g in bbox_groups) + 1),
    ):
        bboxes1, bboxes2 = [b[b[:, -1] == annIdx, :-1] for b in [bboxes1, bboxes2]]
        if not len(bboxes1) > 0 and not len(bboxes2) > 0:
            continue
        if len(bboxes1) > 0 and not len(bboxes2) > 0:
            exclude_list.extend(bboxes1)
        elif not len(bboxes1) > 0 and len(bboxes2) > 0:
            exclude_list.extend(bboxes2)
        else:
            iou = utils.compute_iou(bboxes1, bboxes2)
            exclude_list.extend(bboxes1[~(iou.max(axis=1) > iou_threshold)])
            exclude_list.extend(bboxes2[~(iou.max(axis=0) > iou_threshold)])
    exclude = (
        np.array(exclude_list)
        if len(exclude_list) > 0
        else np.empty((0, 4), dtype="int64")
    )
    include = np.concatenate(
        [
            g[:, :-1] if len(g) > 0 else np.empty((0, 4), dtype="int64")
            for g in bbox_groups
        ],
        axis=0,
    )
    if len(include) > 0:
        include = np.unique(
            include[utils.compute_iou(include, exclude).max(axis=1) == 0]
            if len(exclude) > 0
            else include,
            axis=0,
        )
    return include, exclude


def search(x, y, exclude, include, max_height, max_width, min_height=1, min_width=1):
    """Starting at some coordinates, search for the largest rectangle
    that encompasses include boxes fully and does not cross any exclude boxes."""
    crossings = np.concatenate(
        [
            exclude[
                (
                    (x >= exclude[:, 0])
                    & (x < exclude[:, 2])
                    & (y >= exclude[:, 1])
                    & (y < exclude[:, 3])
                )
            ],
            include[
                (
                    (x > include[:, 0])
                    & (x < include[:, 2])
                    & (y > include[:, 1])
                    & (y < include[:, 3])
                )
            ],
        ],
        axis=0,
    )
    if len(crossings) > 0:
        return (1, crossings[:, 3].max() - y, False)
    xye = exclude[(exclude[:, 2:] > (x, y)).min(axis=1)]
    xyi = include[(include[:, 2:] > (x, y)).min(axis=1)]

    # We can extend as far as (a) the first inclusion box that
    # crosses we would split vertically or (b) the first exclusion
    # box that we would split vertically or (c) the maximum height or
    # (d) the image height.
    dyc_max = int(
        np.concatenate(
            [xyi[xyi[:, 0] < x, 1] - y, xye[xye[:, 0] <= x, 1] - y, [max_height]]
        ).min()
    )
    dx, dy = 0, 0
    for dyc in range(int(min_height), int(dyc_max) + 1):
        # We can go as far as (a) the first exclusion box OR
        # (b) the first inclusion box that crosses at the
        # starting (yc) or current (yc + dyc) y-value or (c)
        # the maximum width for the boxes or (d) the width
        # of the image.
        dxc_max = np.concatenate(
            [
                xye[(xye[:, 1] < (y + dyc)) & (xye[:, 0] != x), 0] - x,
                xyi[
                    ((xyi[:, 1] < (y + dyc)) & (xyi[:, 3] > (y + dyc)))
                    | (xyi[:, 1] < y),
                    0,
                ]
                - x,
                [max_width],
            ]
        ).min()
        # Ranges of x-values that would result in splitting an
        # inclusion box.
        inclusion_ranges = xyi[(xyi[:, 1] < (y + dyc))][:, [0, 2]]
        for dxc in range(int(dxc_max), int(max(min_width - 1, 0)), -1):
            if (
                len(inclusion_ranges) == 0
                or (
                    # The inclusion range starts after this point.
                    (inclusion_ranges[:, 0] >= (x + dxc))
                    # The inclusion range ends before this point.
                    | (inclusion_ranges[:, 1] <= (x + dxc))
                ).all()
            ):
                if (dxc * dyc) >= (dx * dy):
                    dx, dy = dxc, dyc
                break
    return dx, dy, (dx > 0 and dy > 0)


def find_acceptable_crops(
    include: np.ndarray,
    width: int,
    height: int,
    exclude: np.ndarray = None,
    max_width: int = None,
    max_height: int = None,
    cache=None,
) -> np.ndarray:
    """Given a list of consensus and non-consensus regions, crop the image into segments
    that avoid the non-consensus regions while not splitting the consensus regions."""
    if max_width is None:
        max_width = width
    if max_height is None:
        max_height = height
    if exclude is None:
        exclude = np.empty((0, 4))
    key = None
    if cache is not None:
        key = (
            hash(include.tobytes()),
            width,
            height,
            hash(exclude.tobytes()),
            max_width,
            max_height,
        )
        crops = cache.get(key)
        if crops is not None:
            return crops
    crops = []
    yfrontier = np.zeros(width, dtype="int32")
    while True:
        xc1 = typing.cast(int, yfrontier.argmin())
        yc1 = yfrontier[xc1]
        dx1, dy1, success1 = search(
            x=xc1,
            y=yc1,
            exclude=exclude,
            include=include,
            max_height=min(max_height, height - yc1),
            max_width=min(max_width, width - xc1),
        )
        xc2, yc2 = xc1 + dx1, yc1 + dy1
        if (yfrontier[xc1:xc2] == yc2).all():
            # We've made no progress. Stop.
            break
        yfrontier[xc1:xc2] = yfrontier[xc1:xc2].clip(min=yc2)
        if not success1:
            continue
        dx2, dy2, success2 = search(
            x=width - (xc1 + dx1),
            y=height - (yc1 + dy1),
            exclude=(width, height, width, height) - exclude[:, [2, 3, 0, 1]],
            include=(width, height, width, height) - include[:, [2, 3, 0, 1]],
            max_height=min(max_height, yc1 + dy1),
            max_width=min(max_width, xc1 + dx1),
            min_height=dy1,
            min_width=dx1,
        )
        if success2:
            xc1, yc1 = xc2 - dx2, yc2 - dy2
            yfrontier[xc1:xc2] = yfrontier[xc1:xc2].clip(min=yc2)
        crops.append([xc1, yc1, xc2, yc2])
    crops = np.array(crops).round().astype("int32")
    if cache is not None:
        cache[key] = crops
    return crops


def visualize_crops(
    include: np.ndarray,
    crops: np.ndarray,
    width: int,
    height: int,
    exclude: np.ndarray = None,
    canvas: np.ndarray = None,
) -> np.ndarray:
    """Create a visual of the consensus, non-consensus, and a set of crops."""
    if exclude is None:
        exclude = np.empty((0, 4))
    visual = (
        canvas
        if canvas is not None
        else (np.zeros((height, width, 4)) + (0, 0, 0, 255)).astype("uint8")
    )
    if canvas is None:
        # Draw filled regions if it's a blank image.
        for xc1, yc1, xc2, yc2 in crops:
            cv2.rectangle(
                visual,
                pt1=(xc1, yc1),
                pt2=(xc2, yc2),
                thickness=-1,
                color=(0, 0, 255, 120),
            )
    for xc1, yc1, xc2, yc2 in crops:
        cv2.rectangle(
            visual,
            pt1=(xc1, yc1),
            pt2=(xc2, yc2),
            thickness=2,
            color=(255, 255, 0, 255),
        )
    for x1, y1, x2, y2 in include:
        visual[y1:y2, x1:x2] = (0, 255, 0, 255)
    for x1, y1, x2, y2 in exclude:
        visual[y1:y2, x1:x2] = (255, 0, 0, 255)
    return visual
