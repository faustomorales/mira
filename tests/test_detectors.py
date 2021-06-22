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
                    selection=core.Selection([[0, 0], [50, 50]]),
                )
            ],
        ),
        core.Scene(
            image=core.utils.get_blank_image(width=100, height=100, n_channels=3),
            annotation_config=annotation_config,
            annotations=[
                core.Annotation(
                    category=annotation_config["bar"],
                    selection=core.Selection([[20, 20], [80, 80]]),
                )
            ],
        ),
    ],
)


def test_yolo():
    yolo = detectors.YOLOv3(
        pretrained_backbone=False, annotation_config=annotation_config
    )
    inverted = yolo.invert_targets(
        y=yolo.compute_targets(collection, input_shape=input_shape),
        input_shape=input_shape,
        nms_threshold=1.0,
    )
    assert all(
        e - 1 <= a < e + 1
        for a, e in zip(inverted[0][0].selection.bbox(), [0, 0, 50, 50])
    )
    assert all(
        e - 1 <= a < e + 1
        for a, e in zip(inverted[1][0].selection.bbox(), [20, 20, 80, 80])
    )
    # Verify that training doesn't crash.
    history = yolo.train(training=collection, epochs=2, train_shape=input_shape)
