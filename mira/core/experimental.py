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
                    np.product(subgroup[:, 2:] - subgroup[:, :2], axis=1).argmin()
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
    exclude = []
    for (bboxes1, bboxes2), annIdx in itertools.product(
        itertools.combinations(bbox_groups, 2),
        range(max([g[:, -1].max() if len(g) > 0 else 0 for g in bbox_groups])),
    ):
        bboxes1, bboxes2 = [b[b[:, -1] == annIdx, :-1] for b in [bboxes1, bboxes2]]
        if not len(bboxes1) > 0 and not len(bboxes2) > 0:
            continue
        if len(bboxes1) > 0 and not len(bboxes2) > 0:
            exclude.extend(bboxes1)
        elif not len(bboxes1) > 0 and len(bboxes2) > 0:
            exclude.extend(bboxes2)
        else:
            iou = utils.compute_iou(bboxes1, bboxes2)
            exclude.extend(bboxes1[~(iou.max(axis=1) > iou_threshold)])
            exclude.extend(bboxes2[~(iou.max(axis=0) > iou_threshold)])
    exclude = np.array(exclude) if len(exclude) > 0 else np.empty((0, 4))
    include = np.concatenate(
        [g[:, :-1] if len(g) > 0 else np.empty((0, 4)) for g in bbox_groups], axis=0
    )
    if len(include) > 0:
        include = np.unique(
            include[utils.compute_iou(include, exclude).max(axis=1) == 0]
            if len(exclude) > 0
            else include,
            axis=0,
        )
    return include, exclude


def find_consensus_crops(
    include: np.ndarray, exclude: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Given a list of consensus and non-consensus regions, crop the image into segments
    that avoid the non-consensus regions while not splitting the consensus regions."""
    crops = []
    yfrontier = np.zeros(width)
    while True:
        xc1 = yfrontier.argmin()
        yc1 = yfrontier[xc1]
        exclude_frontier = (
            (xc1 >= exclude[:, 0])
            & (xc1 < exclude[:, 2])
            & (yc1 >= exclude[:, 1])
            & (yc1 < exclude[:, 3])
        )
        if exclude_frontier.any():
            yfrontier[xc1] = exclude[exclude_frontier, 3].max()
            continue
        include_frontier = (
            (xc1 >= include[:, 0])
            & (xc1 < include[:, 2])
            & (yc1 >= include[:, 1])
            & (yc1 < include[:, 3])
        )
        if include_frontier.any():
            yfrontier[xc1] = include[include_frontier, 3].max() + 1
            continue
        xye = exclude[(exclude[:, 2:] > (xc1, yc1)).min(axis=1)]
        xyi = include[(include[:, 2:] > (xc1, yc1)).min(axis=1)]
        crossed_inclusion_vertically = xyi[:, 0] < xc1
        dycm = int(
            (
                xyi[crossed_inclusion_vertically, 1].min()
                if crossed_inclusion_vertically.any()
                else height
            )
            - yc1
        )
        dx, dy = 0, 0
        for dyc in range(1, dycm + 1):
            # Exclusion boxes that we would hit at the current
            # y-value.
            crossed_exclusion = xye[:, 1] < (yc1 + dyc)

            # Inclusion boxes that we can never cross because they are split
            # by the starting (yc) or current (yc + dyc) y-value.
            crossed_inclusion = (
                (xyi[:, 1] <= (yc1 + dyc)) & (xyi[:, 3] > (yc1 + dyc))
            ) | (xyi[:, 1] < yc1)
            dxc_max = (
                min(
                    xye[crossed_exclusion, 0].min()
                    if crossed_exclusion.any()
                    else width,
                    xyi[crossed_inclusion, 0].min()
                    if crossed_inclusion.any()
                    else width,
                )
                - xc1
            )

            # Ranges of x-values that would result in splitting an
            # inclusion box.
            inclusion_ranges = xyi[(xyi[:, 1] < (yc1 + dyc))][:, [0, 2]]
            for dxc in range(dxc_max, 0, -1):
                if (
                    len(inclusion_ranges) == 0
                    or (
                        (inclusion_ranges[:, 1] < (xc1 + dxc))
                        | (inclusion_ranges[:, 0] > (xc1 + dxc))
                    ).all()
                ):
                    if (dxc * dyc) > (dx * dy):
                        dx, dy = dxc, dyc
                    break
        xc2, yc2 = xc1 + dx, yc1 + dy
        if (yfrontier[xc1:xc2] == yc2).all():
            # We've made no progress. Stop.
            break
        yfrontier[xc1:xc2] = yfrontier[xc1:xc2].clip(min=yc2)
        crops.append([xc1, yc1, xc2, yc2])
    return np.array(crops).round().astype("int32")


def visualize_consensus_crops(
    include: np.ndarray,
    exclude: np.ndarray,
    crops: np.ndarray,
    width: int,
    height: int,
    canvas: np.ndarray = None,
) -> np.ndarray:
    """Create a visual of the consensus, non-consensus, and a set of crops."""
    visual = (
        canvas
        if canvas is not None
        else np.zeros((height, width, 4), dtype="uint8") + (0, 0, 0, 255)
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
