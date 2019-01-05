from mira import detectors, core


annotation_config = core.AnnotationConfiguration(['foo', 'bar'])
collection = core.SceneCollection(
    annotation_config=annotation_config,
    scenes=[
        core.Scene(
            image=core.Image.new(
                width=100,
                height=100,
                channels=3
            ),
            annotation_config=annotation_config,
            annotations=[
                core.Annotation(
                    category=annotation_config['foo'],
                    selection=core.Selection([[0, 0], [50, 50]])
                )
            ]
        ),
        core.Scene(
            image=core.Image.new(
                width=100,
                height=100,
                channels=3
            ),
            annotation_config=annotation_config,
            annotations=[
                core.Annotation(
                    category=annotation_config['bar'],
                    selection=core.Selection([[20, 20], [80, 80]])
                )
            ]
        ),
    ]
)


def test_retinanet():
    rn = detectors.RetinaNet(
        pretrained_backbone=False,
        annotation_config=annotation_config
    )
    inverted = rn.invert_targets(y=rn.compute_targets(collection), images=collection.images)  # noqa: E501
    assert all(e - 1 <= a < e + 1 for a, e in zip(inverted[0].annotations[0].selection.bbox(), [0, 0, 50, 50]))  # noqa: E501
    assert all(e - 1 <= a < e + 1 for a, e in zip(inverted[1].annotations[0].selection.bbox(), [20, 20, 80, 80]))  # noqa: E501


def test_yolo():
    yolo = detectors.YOLOv3(
        pretrained_backbone=False,
        annotation_config=annotation_config
    )
    inverted = yolo.invert_targets(y=yolo.compute_targets(collection), images=collection.images)  # noqa: E501
    assert all(e - 1 <= a < e + 1 for a, e in zip(inverted[0].annotations[0].selection.bbox(), [0, 0, 50, 50]))  # noqa: E501
    assert all(e - 1 <= a < e + 1 for a, e in zip(inverted[1].annotations[0].selection.bbox(), [20, 20, 80, 80]))  # noqa: E501


def test_east():
    east = detectors.EAST(
        pretrained_backbone=False,
        annotation_config=annotation_config,
        text_category='foo'
    )
    inverted = east.invert_targets(y=east.compute_targets(collection), images=collection.images)  # noqa: E501
    assert all(e - 1 <= a < e + 1 for a, e in zip(inverted[0].annotations[0].selection.bbox(), [0, 0, 50, 50]))  # noqa: E501
    assert len(inverted[1].annotations) == 0


def test_advancedeast():
    advancedeast = detectors.AdvancedEAST(
        pretrained_backbone=False,
        annotation_config=annotation_config,
        text_category='foo'
    )
    inverted = advancedeast.invert_targets(y=advancedeast.compute_targets(collection), images=collection.images)  # noqa: E501
    assert all(e - 2 <= a < e + 2 for a, e in zip(inverted[0].annotations[0].selection.bbox(), [0, 0, 50, 50]))  # noqa: E501
    assert len(inverted[1].annotations) == 0

