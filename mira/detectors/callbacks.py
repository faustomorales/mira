import os
import glob
import json
import typing
import shutil

try:
    import torch
except ImportError:
    torch = None  # type: ignore
import cv2
import numpy as np
import pandas as pd
import typing_extensions as tx

if typing.TYPE_CHECKING:
    from .detector import Detector
from .. import metrics, core

# pylint: disable=too-few-public-methods
class CallbackProtocol(tx.Protocol):
    """A protocol defining how we expect callbacks to behave."""

    def __call__(
        self,
        detector: "Detector",
        summaries: typing.List[typing.Dict[str, typing.Any]],
        data_dir: str,
    ) -> typing.Dict[str, typing.Any]:
        pass


def best_weights(
    filepath, metric="loss", method="min", key="saved"
) -> CallbackProtocol:
    """A callback that saves the best model weights according to a metric.

    Args:
        metric: The metric to track. Use dot notation for nested attributes
            (e.g., val_mAP.{class_name}).
        method: How to handle the metric ("min" minimizes the metric while "max"
            maximizes it).
        key: What name to use for the saved flag.
    """
    # pylint: disable=unused-argument
    def callback(detector, summaries, data_dir=None):
        saved = False
        summaries_df = pd.json_normalize(summaries)
        best_idx = (
            summaries_df[metric].idxmax()
            if method == "max"
            else summaries_df[metric].idxmin()
        )
        if best_idx == len(summaries_df) - 1:
            detector.save_weights(filepath)
            saved = True
        return {key: saved}

    return callback


def csv_logger(filepath) -> CallbackProtocol:
    """A callback that saves a CSV of the summaries to a specific
    filepath.

    Args:
        filepath: The filepath where the logs will be saved.
    """
    # pylint: disable=unused-argument
    def callback(detector, summaries, data_dir):
        pd.json_normalize(summaries).to_csv(filepath, index=False)
        return {}

    return callback


def load_json(filepath: str):
    """Load JSON from file"""
    with open(filepath, "r", encoding="utf8") as f:
        return json.loads(f.read())


def data_dir_to_collections(data_dir: str, threshold: float, detector: "Detector"):
    """Convert a temporary training artifact directory into a set
    of train and validation (if present) true/predicted collections."""
    return {
        split: {
            "collections": {
                "true_collection": core.SceneCollection(
                    [
                        core.Scene(
                            image=filepath,
                            categories=detector.categories,
                            annotations=[
                                core.Annotation(
                                    detector.categories[cIdx], x1, y1, x2, y2
                                )
                                for x1, y1, x2, y2, cIdx in np.load(
                                    filepath + ".bboxes.npz"
                                )["bboxes"]
                            ],
                        )
                        for filepath in images
                    ],
                    categories=detector.categories,
                ),
                "pred_collection": core.SceneCollection(
                    [
                        core.Scene(
                            image=filepath,
                            categories=detector.categories,
                            annotations=annotations,
                        )
                        for filepath, annotations in zip(
                            images,
                            detector.invert_targets(
                                {
                                    "output": [
                                        {
                                            k: torch.Tensor(v)
                                            for k, v in np.load(
                                                filepath + ".output.npz"
                                            ).items()
                                        }
                                        for filepath in images
                                    ]
                                },
                                threshold=threshold,
                            ),
                        )
                    ],
                    categories=detector.categories,
                ),
            },
            "transforms": transforms,
            "metadata": pd.json_normalize(metadatas),
            "indices": [
                int(os.path.basename(filepath).split(".")[0]) for filepath in images
            ],
        }
        for split, images, metadatas, transforms in [
            (
                split,
                images,
                [load_json(f + ".metadata.json") for f in images],
                [np.load(f + ".transform.npz")["transform"] for f in images],
            )
            for split, images in [
                (
                    split,
                    glob.glob(os.path.join(data_dir, split, "*.png"))
                    + glob.glob(os.path.join(data_dir, split, "*", "*.png")),
                )
                for split in ["train", "val"]
            ]
        ]
        if len(images) > 0
    }


def mAP(iou_threshold=0.5, threshold=0.01) -> CallbackProtocol:
    """Build a callback that computes mAP. All kwargs
    passed to detector.mAP()"""

    # pylint: disable=unused-argument
    def callback(detector, summaries, data_dir):
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
            for split, split_data in data_dir_to_collections(
                data_dir, threshold=threshold, detector=detector
            ).items()
        }

    return callback


def error_examples(
    examples_dir: str,
    threshold=0.5,
    iou_threshold=0.5,
    pad=10,
    max_crops_per_image=10,
    overwrite=False,
) -> CallbackProtocol:
    """Build a callback that saves errors to an output directory."""
    if os.path.isdir(examples_dir):
        if overwrite:
            shutil.rmtree(examples_dir, ignore_errors=True)
        else:
            raise ValueError(
                "Directory already exists. Delete it or set overwrite=True."
            )

    def callback(detector, summaries, data_dir):
        counts: typing.Dict[str, int] = {}
        entries: typing.List[dict] = []
        for split, split_data in data_dir_to_collections(
            data_dir, threshold=threshold, detector=detector
        ).items():
            for imageIdx, image, examples, metadata, transform in zip(
                split_data["indices"],
                split_data["collections"]["true_collection"].images,
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
                    bboxes = core.utils.transform_bboxes(
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
                    for annIdx, annotation in enumerate(
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
                            annotation.extract(image, pad=pad)[..., ::-1],
                        )
        pd.DataFrame(entries).to_csv(
            os.path.join(examples_dir, f"{len(summaries)}.csv"), index=False
        )
        return counts

    return callback
