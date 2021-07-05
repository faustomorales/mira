import torch
import pandas as pd


def best_weights(filepath, metric="loss", method="min"):
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


def csv_logger(filepath):
    """A callback that saves a CSV of the summaries to a specific
    filepath.

    Args:
        filepath: The filepath where the logs will be saved.
    """

    def callback(_, summaries):
        pd.json_normalize(summaries).to_csv(filepath, index=False)
        return {}

    return callback
