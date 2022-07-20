# pylint: disable=invalid-name
"""Metrics for object detection tasks."""
import typing

import numpy as np

from .core import SceneCollection, Annotation, utils

# pylint: disable=unsubscriptable-object
def precision_recall_curve(
    true_collection: SceneCollection,
    pred_collection: SceneCollection,
    iou_threshold: float = 0.5,
) -> typing.Dict[str, np.ndarray]:
    """Compute the precision-recall curve for each of the
    classes.

    Args:
        true_collection: The true scene collection
        pred_collection: The predicted scene collection
        iou_threshold: The threshold for detection

    Returns:
        A dict with category names as keys and array of shape (Ni, 3)
        which is the precision, recall, and score for each of the
        predicted boxes for the category.
    """
    assert (
        true_collection.categories == pred_collection.categories
    ), "Annotation configurations must match"
    categories = true_collection.categories
    assert len(true_collection.scenes) == len(
        pred_collection.scenes
    ), "Must have same scenes in each collection"

    # The ith entry in tfs is a list of lists, each of length three,
    # which are the change in the number of true positives and
    # false positives, along with the score at which the change
    # occurred for the ith class.
    tfs: typing.List[typing.List[typing.List[int]]] = [
        [[], [], []] for c in range(len(categories))
    ]

    # The ith entry in tfs is the number of true boxes
    # for the ith class.
    pos = [0 for c in range(len(categories))]

    for true, pred in zip(true_collection, pred_collection):
        pred_bboxes = pred.bboxes()
        true_bboxes = true.bboxes()
        pred_scores = pred.scores()
        assert all(
            s is not None for s in pred_scores
        ), "All annotations must have a score."

        for classIdx in range(len(categories)):
            pred_bboxes_cur = pred_bboxes[pred_bboxes[:, 4] == classIdx]
            true_bboxes_cur = true_bboxes[true_bboxes[:, 4] == classIdx]
            pred_scores_cur = pred_scores[pred_bboxes[:, 4] == classIdx]

            nPredicted = len(pred_bboxes_cur)
            nTrue = len(true_bboxes_cur)
            pos[classIdx] += nTrue

            if nPredicted == 0:
                # We have no new information to add if there were no
                # predicted boxes.
                continue

            if nTrue == 0:
                # All of them are false positives
                for score in pred_scores_cur:
                    tfs[classIdx][0].append(0)
                    tfs[classIdx][1].append(1)
                    tfs[classIdx][2].append(score)
                continue

            # Sort the predicted boxes by decreasing confidence
            pred_bboxes_cur = pred_bboxes_cur[(-pred_scores_cur).argsort()]
            pred_scores_cur = pred_scores_cur[(-pred_scores_cur).argsort()]

            # (n, m): status for ith prediction for jth true box
            det = (
                utils.compute_iou(
                    pred_bboxes_cur[:, :4],
                    true_bboxes_cur[:, :4],
                )
                > iou_threshold
            )

            fp_prev = 0
            tp_prev = 0
            for i in range(nPredicted):
                tp_cur = det[: i + 1].max(axis=0).sum()
                fp_cur = (i + 1) - det[: i + 1].max(axis=1).sum()

                tp_delta = tp_cur - tp_prev
                fp_delta = fp_cur - fp_prev

                assert tp_delta >= 0
                assert fp_delta >= 0
                assert tp_cur <= nTrue
                assert fp_cur <= nPredicted

                tp_prev = tp_cur
                fp_prev = fp_cur

                tfs[classIdx][0].append(tp_delta)
                tfs[classIdx][1].append(fp_delta)
                tfs[classIdx][2].append(pred_scores_cur[i])

    prs = [None for n in range(len(categories))]

    for classIdx, tfs_cur, pos_cur in zip(range(len(categories)), tfs, pos):
        # If we had no detections AND there
        # were no true boxes, precision and recall
        # are not defined.
        tfs_cur_arr = np.array(tfs_cur).T
        tfs_cur_arr = tfs_cur_arr[(-tfs_cur_arr[:, 2]).argsort()]
        tp = tfs_cur_arr[:, 0].cumsum()
        fp = tfs_cur_arr[:, 1].cumsum()
        scores = tfs_cur_arr[:, 2]

        precisions = tp / (tp + fp)
        recalls = tp / pos_cur
        prs[classIdx] = np.vstack([precisions, recalls, scores]).T  # type: ignore
    return dict(zip([c.name for c in categories], prs))  # type: ignore


def mAP(
    true_collection: SceneCollection,
    pred_collection: SceneCollection,
    iou_threshold: float = 0.5,
) -> typing.Dict[str, float]:
    """Compute mAP (mean average precision) for
    a pair of scene collections.

    Args:
        true_collection: The true scene collection
        pred_collection: The predicted scene collection
        iou_threshold: The threshold for detection

    Returns:
        mAP class scores
    """
    prs = precision_recall_curve(true_collection, pred_collection, iou_threshold)
    aps = {}
    for className, prs_cur in prs.items():
        ps = prs_cur[:, 0]
        rs = prs_cur[:, 1].astype("float32")
        pi = np.zeros(11)
        # If rs is None, there were no detections and no true
        # boxes. If it is all nans, then there were no
        # true boxes but there were detections.
        if rs is None or np.isnan(rs).sum() == rs.shape[0]:
            aps[className] = np.nan
            continue
        for i, r in enumerate(np.linspace(0, 1, 11)):
            # From section 4.2 of VOC paper,the precision at each
            # recall level r is interpolated by taking the maximum
            # precision measured for a method for which
            # the corresponding recall exceeds r
            pc = ps[rs >= r]
            if len(pc) > 0:  # pylint: disable=len-as-condition
                pi[i] = pc.max()
        aps[className] = pi.mean()
    return aps


def crop_error_examples(
    true_collection: SceneCollection,
    pred_collection: SceneCollection,
    threshold=0.3,
    iou_threshold=0.1,
) -> typing.List[typing.Dict[str, typing.List[Annotation]]]:
    """Get crops of true positives, false negatives, and false positives.
    Args:
        true_collection: A collection of the ground truth scenes.
        pred_collection: A collection of the predicted scenes.
        threshold: The score threshold for selecting annotations from predicted
            scenes.
        iou_threhsold: The IoU threshold for counting a box as a true positive.
    Returns:
        A list of dicts with "tps", "fps", and "fns"
        with the same length of the input collections. The values in each dict
        are crops from the original image.
    """
    examples = []
    for true_scene, pred_scene in zip(true_collection, pred_collection):
        pred_scene = pred_scene.assign(
            annotations=[
                a
                for a in pred_scene.annotations
                if a.score is None or a.score > threshold
            ]
        )
        boxes_true = true_scene.bboxes()[:, :4]
        boxes_pred = pred_scene.bboxes()[:, :4]
        iou = utils.compute_iou(boxes_pred, boxes_true)
        examples.append(
            {
                "tps": [
                    ann.assign(score=pred_scene.annotations[predIdx].score)
                    for ann, iou, predIdx in zip(
                        true_scene.annotations, iou.max(axis=0), iou.argmax(axis=0)
                    )
                    if iou > iou_threshold
                ]
                if (pred_scene.annotations and true_scene.annotations)
                else [],
                "fps": [
                    ann
                    for ann, iou in zip(
                        pred_scene.annotations,
                        iou.max(axis=1)
                        if true_scene.annotations
                        else [-1] * len(pred_scene.annotations),
                    )
                    if iou < iou_threshold
                ]
                if len(pred_scene.annotations) > 0
                else [],
                "fns": [
                    ann
                    for ann, iou in zip(
                        true_scene.annotations,
                        iou.max(axis=0)
                        if pred_scene.annotations
                        else [-1] * len(true_scene.annotations),
                    )
                    if iou < iou_threshold
                ]
                if true_scene.annotations
                else [],
            }
        )
    return examples
