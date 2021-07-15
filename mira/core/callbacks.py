import typing

import torch
import pandas as pd
import typing_extensions as tx

if typing.TYPE_CHECKING:
    from ..detectors import Detector

# pylint: disable=too-few-public-methods
class CallbackProtocol(tx.Protocol):
    """A protocol defining how we expect callbacks to behave."""

    def __call__(
        self, detector: "Detector", summaries: typing.List[typing.Dict[str, typing.Any]]
    ) -> typing.Dict[str, typing.Any]:
        pass


def best_weights(filepath, metric="loss", method="min") -> CallbackProtocol:
    """A callback that saves the best model weights according to a metric.

    Args:
        metric: The metric to track. Use dot notation for nested attributes
            (e.g., val_mAP.{class_name}).
        method: How to handle the metric ("min" minimizes the metric while "max"
            maximizes it).
    """

    def callback(detector, summaries):
        saved = False
        summaries_df = pd.json_normalize(summaries)
        best_idx = (
            summaries_df[metric].idxmax()
            if method == "max"
            else summaries_df[metric].idxmin()
        )
        if best_idx == len(summaries_df) - 1:
            torch.save(detector.model.state_dict(), filepath)
            saved = True
        return {"saved": saved}

    return callback


def csv_logger(filepath) -> CallbackProtocol:
    """A callback that saves a CSV of the summaries to a specific
    filepath.

    Args:
        filepath: The filepath where the logs will be saved.
    """
    # pylint: disable=unused-argument
    def callback(detector, summaries):
        pd.json_normalize(summaries).to_csv(filepath, index=False)
        return {}

    return callback


def mAP(collection, key="val_mAP", **kwargs) -> CallbackProtocol:
    """Build a callback that computes mAP. All kwargs
    passed to detector.mAP()"""

    # pylint: disable=unused-argument
    def callback(detector, summaries):
        return {
            key: {
                k: round(v, 2)
                for k, v in detector.mAP(collection=collection, **kwargs).items()
            }
        }

    return callback
