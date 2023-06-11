import os
import typing
import shutil

import cv2
import numpy as np
import pandas as pd
from . import torchtools, utils
from .. import metrics


def best_weights(
    filepath, metric="loss", method="min", key="saved"
) -> torchtools.CallbackProtocol:
    """A callback that saves the best model weights according to a metric.

    Args:
        metric: The metric to track. Use dot notation for nested attributes
            (e.g., val_mAP.{class_name}).
        method: How to handle the metric ("min" minimizes the metric while "max"
            maximizes it).
        key: What name to use for the saved flag.
    """

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        saved = False
        summaries_df = pd.json_normalize(summaries)
        best_idx = (
            summaries_df[metric].idxmax()
            if method == "max"
            else summaries_df[metric].idxmin()
        )
        if best_idx == len(summaries_df) - 1:
            model.save_weights(filepath)
            saved = True
        return {key: saved}

    return callback


def csv_logger(filepath) -> torchtools.CallbackProtocol:
    """A callback that saves a CSV of the summaries to a specific
    filepath.

    Args:
        filepath: The filepath where the logs will be saved.
    """

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        pd.json_normalize(summaries).to_csv(filepath, index=False)
        return {}

    return callback


def mAP(iou_threshold=0.5) -> torchtools.CallbackProtocol:
    """Build a callback that computes mAP. All kwargs
    passed to metrics.mAP()"""

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        return {
            f"{split}_mAP": round(
                np.nanmean(
                    list(
                        metrics.mAP(
                            **split_data["collections"],
                            iou_threshold=iou_threshold,
                        ).values()
                    )
                ),
                2,
            )
            for split, split_data in collections.items()
        }

    return callback


def mIOU(**kwargs) -> torchtools.CallbackProtocol:
    """Build a callback that computes mIOU."""

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        return {
            f"{split}_mIOU": round(
                np.nanmean(
                    list(metrics.mIOU(**split_data["collections"], **kwargs).values())
                ),
                2,
            )
            for split, split_data in collections.items()
        }

    return callback


def error_examples(
    examples_dir: str,
    threshold=0.5,
    iou_threshold=0.5,
    pad=10,
    max_crops_per_image=10,
    overwrite=False,
) -> torchtools.CallbackProtocol:
    """Build a callback that saves errors to an output directory."""
    if os.path.isdir(examples_dir):
        if overwrite:
            shutil.rmtree(examples_dir, ignore_errors=True)
        else:
            raise ValueError(
                "Directory already exists. Delete it or set overwrite=True."
            )

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        counts: typing.Dict[str, int] = {}
        entries: typing.List[dict] = []
        for split, split_data in collections.items():
            for imageIdx, image, examples, metadata, transform in zip(
                split_data["indices"],
                split_data["collections"]["true_collection"].images(),
                metrics.crop_error_examples(
                    **split_data["collections"],
                    threshold=threshold,
                    iou_threshold=iou_threshold,
                ),
                split_data["metadata"].to_dict(orient="records")
                or [{}] * len(split_data["metadata"]),
                split_data["transforms"],
            ):
                for error_type, annotations in examples.items():
                    directory = os.path.join(
                        examples_dir, str(len(summaries)), split, error_type
                    )
                    bboxes = utils.transform_bboxes(
                        np.array([ann.x1y1x2y2() for ann in annotations]),
                        np.linalg.pinv(transform),
                        clip=False,
                    )
                    entries.extend(
                        [
                            {
                                **metadata,
                                "split": split,
                                "imageIdx": imageIdx,
                                "annIdx": annIdx,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "score": ann.score,
                                "error_type": error_type,
                                "category": ann.category.name,
                            }
                            for annIdx, (ann, (x1, y1, x2, y2)) in enumerate(
                                zip(
                                    annotations,
                                    bboxes,
                                )
                            )
                        ]
                    )
                    os.makedirs(directory, exist_ok=True)
                    counts[error_type] = counts.get(error_type, 0) + len(annotations)
                    for annIdx, ann in enumerate(
                        sorted(
                            annotations,
                            key=lambda ann: ann.score
                            if ann.score is not None
                            else ann.area(),
                            reverse=True,
                        )
                    ):
                        if annIdx >= max_crops_per_image:
                            break
                        cv2.imwrite(
                            os.path.join(directory, f"{imageIdx}_{annIdx}.png"),
                            ann.extract(image, pad=pad)[..., ::-1],
                        )
        pd.DataFrame(entries).to_csv(
            os.path.join(examples_dir, f"{len(summaries)}.csv"), index=False
        )
        return counts

    return callback


def classification_metrics(**kwargs) -> torchtools.CallbackProtocol:
    """Returns classification metrics (precision, recall, f1). All arguments passed to metrics.classification_metrics"""

    # pylint: disable=unused-argument
    def callback(model, summaries, collections):
        return dict(
            utils.flatten(
                [
                    utils.flatten(
                        [
                            [
                                (
                                    f"{split}_{category}_{metric_name}",
                                    round(metric_value, 2)
                                    if np.isfinite(metric_value)
                                    else np.nan,
                                )
                                for metric_name, metric_value in scores.items()
                            ]
                            for category, scores in metrics.classification_metrics(
                                **split_data["collections"], **kwargs
                            ).items()
                        ]
                    )
                    for split, split_data in collections.items()
                ]
            )
        )

    return callback
