from typing import List, Tuple

import numpy as np

from ..utils import compute_overlap
from ..core import SceneCollection


def precision_recall_curve(
    true_collection: SceneCollection,
    pred_collection: SceneCollection,
    iou_threshold: float=0.5
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the precision-recall curve for each of the
    classes.

    Args:
        true_collection: The true scene collection
        pred_collection: The predicted scene collection
        iou_threshold: The threshold for detection

    Returns:
        A list of 3-tuples, one for each class. The tuple
        consists of the precision, recall, and score
        thresholds, respectively.
    """
    assert true_collection.annotation_config == pred_collection.annotation_config, (  # noqa: E501
        'Annotation configurations must match'
    )
    annotation_config = true_collection.annotation_config
    assert len(true_collection.scenes) == len(pred_collection.scenes), \
        'Must have same scenes in each collection'

    # The ith entry in tfs is a list of length three,
    # which are the change in the number of true positives and
    # false positives, along with the score at which the change
    # occurred for the ith class.
    tfs = [[] for c in range(len(annotation_config))]

    # The ith entry in tfs is the number of true boxes
    # for the ith class.
    pos = [0 for c in range(len(annotation_config))]

    for true, pred in zip(true_collection, pred_collection):
        pred_bboxes = pred.bboxes()
        true_bboxes = true.bboxes()
        pred_scores = pred.scores()

        for c in range(len(annotation_config)):
            pred_bboxes_cur = pred_bboxes[pred_bboxes[:, 4] == c]
            true_bboxes_cur = true_bboxes[true_bboxes[:, 4] == c]
            n = len(pred_bboxes_cur)
            m = len(true_bboxes_cur)
            pos[c] += m

            if n == 0 and m == 0:
                continue

            # (n, m): The IoU between the ith predicted
            #         box and the jth true box
            det = compute_overlap(
                boxes=pred_bboxes_cur.astype('float64'),
                query_boxes=true_bboxes_cur.astype('float64')
            ) > iou_threshold

            # (n): The index of the ith highest confidence
            #      prediction
            pred_bbox_idx = (-pred_scores).argsort()
            tp = 0
            fp = 0
            for i in range(n):
                # For each of the best i bounding boxes,
                # find the number of true ground boxes that
                # were detected.
                tpc = det[pred_bbox_idx[:i+1], :].max(axis=0).sum()
                fpc = (i + 1) - tpc
                sc = pred_scores[pred_bbox_idx[i]]

                tpd = tpc - tp
                fpd = fpc - fp

                tp = tpc
                fp = fpc

                # We keep a running delta for
                # later use.
                tfs[c].append([tpd, fpd, sc])

    prs = [(None, None, None)]*len(annotation_config)

    for tfs_cur, pos_cur, c in zip(
        tfs, pos, range(len(annotation_config))
    ):
        # If we had no detections or there
        # were no true boxes, precision and recall
        # are not defined.
        if len(tfs_cur) == 0 or pos_cur == 0:
            continue
        tfs_cur = np.array(tfs_cur)
        tfs_cur = tfs_cur[(-tfs_cur[:, 2]).argsort()]

        tp = tfs_cur[:, 0].cumsum()
        fp = tfs_cur[:, 1].cumsum()
        prs[c] = (
            tp / (tp + fp),
            tp / pos_cur,
            tfs_cur[:, 2]
        )
    return prs


def mAP(
    true_collection: SceneCollection,
    pred_collection: SceneCollection,
    iou_threshold: float=0.5
) -> float:
    """Compute mAP (mean average precision) for
    a pair of scene collections.

    Args:
        true_collection: The true scene collection
        pred_collection: The predicted scene collection
        iou_threshold: The threshold for detection

    Returns:
        mAP score
    """
    prs = precision_recall_curve(
        true_collection,
        pred_collection,
        iou_threshold
    )
    aps = np.zeros(len(prs)) * np.nan
    for c, (ps, rs, _) in enumerate(prs):
        pi = np.zeros(11)
        for i, r in enumerate(np.linspace(0, 1, 11)):
            # From section 4.2 of VOC paper,the precision at each
            # recall level r is interpolated by taking the maximum
            # precision measured for a method for which
            # the corresponding recall exceeds r
            pc = ps[rs > r]
            if len(pc) > 0:
                pi[i] = pc.max()
        aps[c] = pi.mean()
    return aps.mean()