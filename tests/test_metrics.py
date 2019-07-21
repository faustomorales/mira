"""Test mAP calculation."""
# pylint: disable=invalid-name

import numpy as np

from mira import core, metrics


def test_mAP():
    """Make sure mAP makes sense."""
    image = np.random.randint(
        low=0, high=255, size=(100, 100, 3)).astype('uint8')

    annotation_config = core.AnnotationConfiguration(['foo', 'bar', 'baz'])
    true = core.Scene(
        image=image,
        annotation_config=annotation_config,
        annotations=[
            core.Annotation(
                selection=core.Selection([[0, 0], [10, 10]]),
                category=annotation_config['foo']),
            core.Annotation(
                selection=core.Selection([[40, 40], [50, 50]]),
                category=annotation_config['baz']),
        ])
    pred = core.Scene(
        image=image,
        annotation_config=annotation_config,
        annotations=[
            core.Annotation(
                selection=core.Selection([[0, 0], [10, 10]]),
                category=annotation_config['foo'],
                score=0.9),
            core.Annotation(
                selection=core.Selection([[15, 15], [30, 30]]),
                category=annotation_config['foo'],
                score=0.5),
            core.Annotation(
                selection=core.Selection([[45, 45], [50, 50]]),
                category=annotation_config['baz'],
                score=0.8)
        ])

    true_collection = core.SceneCollection([true],
                                           annotation_config=annotation_config)
    pred_collection = core.SceneCollection([pred],
                                           annotation_config=annotation_config)

    maps1 = metrics.mAP(true_collection, pred_collection, iou_threshold=0.25)
    maps2 = metrics.mAP(true_collection, pred_collection, iou_threshold=0.3)

    assert maps1['foo'] == 1
    assert maps2['foo'] == 1
    assert ~np.isfinite(maps1['bar'])
    assert ~np.isfinite(maps2['bar'])
    assert maps1['baz'] == 1  # The IoU is just enough
    assert maps2['baz'] == 0  # The IoU is too small
