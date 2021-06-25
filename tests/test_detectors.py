import pytest

from mira import detectors, core

input_shape = (224, 224, 3)
annotation_config = core.AnnotationConfiguration(["foo", "bar"])
collection = core.SceneCollection(
    annotation_config=annotation_config,
    scenes=[
        core.Scene(
            image=core.utils.get_blank_image(width=100, height=100, n_channels=3),
            annotation_config=annotation_config,
            annotations=[
                core.Annotation(
                    category=annotation_config["foo"],
                    selection=core.Selection(0, 0, 50, 50),
                )
            ],
        ),
        core.Scene(
            image=core.utils.get_blank_image(width=100, height=100, n_channels=3),
            annotation_config=annotation_config,
            annotations=[
                core.Annotation(
                    category=annotation_config["bar"],
                    selection=core.Selection(20, 20, 80, 80),
                )
            ],
        ),
    ],
)


@pytest.mark.parametrize("detector_class", [detectors.YOLOv3, detectors.EfficientDet])
def test_basics(detector_class):
    detector = detector_class(
        pretrained_backbone=False,
        annotation_config=annotation_config,
        input_shape=input_shape,
    )
    inverted = detector.invert_targets(
        y=detector.compute_targets(
            collection.annotation_groups, input_shape=input_shape
        ),
        input_shape=input_shape,
    )
    assert all(
        e - 1 <= a < e + 1
        for a, e in zip(inverted[0][0].selection.x1y1x2y2(), [0, 0, 50, 50])
    )
    assert all(
        e - 1 <= a < e + 1
        for a, e in zip(inverted[1][0].selection.x1y1x2y2(), [20, 20, 80, 80])
    )
    # Verify that training doesn't crash.
    detector.train(training=collection, epochs=2, train_shape=input_shape)
