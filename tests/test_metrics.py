"""Test mAP calculation."""
# pylint: disable=invalid-name

import numpy as np

from mira import core, metrics


def test_mAP():
    """Make sure mAP makes sense."""
    image = np.random.randint(low=0, high=255, size=(100, 100, 3)).astype("uint8")

    categories = core.Categories(["foo", "bar", "baz"])
    true = core.Scene(
        image=image,
        categories=categories,
        annotations=[
            core.Annotation(
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                category=categories["foo"],
            ),
            core.Annotation(
                x1=40,
                y1=40,
                x2=50,
                y2=50,
                category=categories["baz"],
            ),
        ],
    )
    pred = core.Scene(
        image=image,
        categories=categories,
        annotations=[
            core.Annotation(
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                category=categories["foo"],
                score=0.9,
            ),
            core.Annotation(
                x1=15,
                y1=15,
                x2=30,
                y2=30,
                category=categories["foo"],
                score=0.5,
            ),
            core.Annotation(
                x1=45,
                y1=45,
                x2=50,
                y2=50,
                category=categories["baz"],
                score=0.8,
            ),
        ],
    )

    true_collection = core.SceneCollection([true], categories=categories)
    pred_collection = core.SceneCollection([pred], categories=categories)

    maps1 = metrics.mAP(true_collection, pred_collection, iou_threshold=0.2)
    maps2 = metrics.mAP(true_collection, pred_collection, iou_threshold=0.3)

    assert maps1["foo"] == 1
    assert maps2["foo"] == 1
    assert ~np.isfinite(maps1["bar"])
    assert ~np.isfinite(maps2["bar"])
    assert maps1["baz"] == 1  # The IoU is just enough
    assert maps2["baz"] == 0  # The IoU is too small


def test_classification():
    image = np.random.randint(low=0, high=255, size=(100, 100, 3)).astype("uint8")
    categories = core.Categories(["foo", "bar", "baz"])
    true = core.SceneCollection(
        [
            core.Scene(
                image=image,
                categories=categories,
                labels=[core.Label(categories["foo"])],
            ),
            core.Scene(
                image=image,
                categories=categories,
                labels=[core.Label(categories["bar"])],
            ),
        ]
    )
    pred = core.SceneCollection(
        [
            core.Scene(
                image=image,
                categories=categories,
                labels=[core.Label(categories["baz"], score=0.5)],
            ),
            core.Scene(
                image=image,
                categories=categories,
                labels=[core.Label(categories["bar"], score=0.5)],
            ),
        ]
    )
    scores = metrics.classification_metrics(true_collection=true, pred_collection=pred)
    assert scores["bar"]["recall"] == 1
    assert scores["baz"]["precision"] == 0
    assert scores["foo"]["recall"] == 0
