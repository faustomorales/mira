import cv2
import pytest
import numpy as np

import mira.core as mc
import mira.detectors as md
import mira.datasets as mds


@pytest.mark.parametrize(
    "detector_class",
    [md.RetinaNet, md.FasterRCNN],
)
def test_detector_edge_cases(detector_class):
    dataset = mds.load_shapes(width=256, height=256, n_scenes=1)
    base = dataset[0]
    dataset = dataset.assign(
        scenes=[
            base,  # A regular case
            base.assign(
                annotations=[], image=np.ones_like(base.image) * 255
            ),  # No annotations
            base.assign(  # The whole image is an entity.
                annotations=[
                    mc.Annotation(
                        x1=0,
                        y1=0,
                        x2=base.image.shape[1],
                        y2=base.image.shape[0],
                        category=dataset.categories["blue circle"],
                    )
                ],
                image=cv2.circle(  # A full screen annotation.
                    np.ones_like(base.image) * 255,
                    center=(base.image.shape[1] // 2, base.image.shape[0] // 2),
                    thickness=-1,
                    color=(0, 0, 255),
                    radius=base.image.shape[0] // 2,
                ),
            ),
            base.assign(  # Very tiny annotations.
                annotations=[
                    mc.Annotation(
                        x1=base.image.shape[1] // 2 - 1,
                        y1=base.image.shape[0] // 2 - 1,
                        x2=base.image.shape[1] // 2 + 1,
                        y2=base.image.shape[0] // 2 + 1,
                        category=dataset.categories["blue circle"],
                    )
                ],
                image=cv2.circle(  # A very small circle.
                    np.ones_like(base.image) * 255,
                    center=(base.image.shape[1] // 2, base.image.shape[0] // 2),
                    thickness=-1,
                    color=(0, 0, 255),
                    radius=1,
                ),
            ),
            base.assign(  # No negatives anywhere. WARNING: This is a nonsense test case -- not actually good for training.
                annotations=[
                    mc.Annotation(
                        x1=0,
                        y1=0,
                        x2=base.image.shape[1],
                        y2=base.image.shape[0],
                        category=c,
                    )
                    for c in dataset.categories
                ]
            ),
        ]
    )
    detector = detector_class(pretrained_backbone=False, categories=dataset.categories)
    detector.model.train()

    # Verify that we can handle a batch of all edge cases.
    detector.loss(dataset)
    for scene in dataset:
        # Verify that we can handle each edge case on its own.
        detector.loss(dataset.assign(scenes=[scene]))
