"""
Test core functions.
"""
import typing
import itertools
import collections
import tempfile
import os
import io

import cv2
import numpy as np

from mira.thirdparty.albumentations import albumentations as A
import mira.core as mc
import mira.core.experimental as mce
import mira.datasets as mds


def test_blank_and_properties():
    """Make sure creating new images works."""
    image = mc.utils.get_blank_image(width=200, height=100, n_channels=2, cval=125)
    assert image.shape == (100, 200, 2)
    assert (image[:, :, :] == 125).all()


def test_scene_deduplication():
    ac = mc.Categories(["foo", "bar"])
    scene = mc.Scene(
        image=mc.utils.get_blank_image(width=200, height=100, n_channels=2, cval=125),
        annotations=[
            mc.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                metadata={"tag": "drop-me-coverage"},
            ),
            mc.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=20,
                y2=20,
                metadata={"tag": "drop-me-always"},
            ),
            mc.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=20,
                y2=20,
                metadata={"tag": "keep-me"},
            ),
            mc.Annotation(
                category=ac["bar"],
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                metadata={"tag": "keep-me"},
            ),
        ],
        categories=ac,
    )
    for convert_to_polygon in [False, True]:
        test_scene = (
            scene.assign(
                annotations=[
                    ann.assign(points=ann.points, x1=None, y1=None, x2=None, y2=None)
                    for ann in scene.annotations
                ]
            )
            if convert_to_polygon
            else scene
        )
        deduplicated_coverage = test_scene.drop_duplicates(method="coverage")
        deduplicated_iou = test_scene.drop_duplicates(method="iou")
        assert len(deduplicated_coverage.annotations) == 2
        assert len(deduplicated_iou.annotations) == 3
        assert all(
            ann.metadata["tag"] == "keep-me"
            for ann in deduplicated_coverage.annotations
        )
        assert all(
            ann.metadata["tag"] in ["keep-me", "drop-me-coverage"]
            for ann in deduplicated_iou.annotations
        )


def test_file_read():
    """Make sure reading files works."""
    image = mc.utils.get_blank_image(width=200, height=100, n_channels=3, cval=125)
    image[40:60, 40:60, 0] = 0
    with tempfile.TemporaryDirectory() as tempdir:
        fpath = os.path.join(tempdir, "test.png")
        mc.utils.save(image, fpath)
        np.testing.assert_allclose(mc.utils.read(fpath), image)
        with open(fpath, "rb") as buffer:
            np.testing.assert_allclose(mc.utils.read(buffer), image)
        with io.BytesIO() as buffer:
            mc.utils.save(image, buffer, ".png")
            buffer.seek(0)
            np.testing.assert_allclose(mc.utils.read(buffer), image)


def test_split():
    n = 5000
    items = np.arange(n).tolist()
    sizes = [0.5, 0.30, 0.20]
    group: typing.List[int] = np.random.choice(500, size=n).tolist()
    stratify: typing.List[int] = np.random.choice(2, size=n).tolist()

    splits1A = mc.utils.split(
        items=items, sizes=sizes, stratify=stratify, group=group, random_state=42
    )

    splits1B = mc.utils.split(
        items=items, sizes=sizes, stratify=stratify, group=group, random_state=42
    )

    splits2 = mc.utils.split(
        items=items, sizes=sizes, stratify=stratify, group=group, random_state=10
    )

    splits = splits1A

    # Given the same random state, the splits should be the same.
    assert splits1A == splits1B

    # Given a different random state, the splits should be different.
    assert splits1A != splits2

    # No overlap between groups.
    assert all(
        len(a.intersection(b)) == 0
        for a, b in itertools.combinations(
            [set(group[idx] for idx in s) for s in splits], 2
        )
    )

    # Roughly match the desired sizes.
    assert all(
        [
            abs(size - len(split) / len(items)) < 0.05
            for split, size in zip(splits, sizes)
        ]
    ), f"Sizes were {splits}"

    # Roughly achieve stratification
    assert all(
        [
            all(
                [
                    abs(size - sum(stratify[i] == key for i in split) / count) < 0.05
                    for size, split in zip(sizes, splits)
                ]
            )
            for key, count in collections.Counter(stratify).items()
        ]
    )


def test_find_consensus_crops():
    width = 512
    height = 768
    for scene in mds.load_shapes(width=width, height=height, n_scenes=10):
        boxes = scene.bboxes()[:, :4]
        include = boxes[:-2]
        exclude = boxes[-2:]
        if (
            len(include) == 0
            or len(exclude) == 0
            or mc.utils.compute_iou(include, exclude).max() > 0
        ):
            continue
        crops = mce.find_acceptable_crops(
            include=include, width=width, height=height, exclude=exclude
        )
        include_coverage = mc.utils.compute_coverage(include, crops)
        exclude_coverage = mc.utils.compute_coverage(exclude, crops)
        # Crops never include an exclusion box.
        assert (exclude_coverage == 0).all()
        # Crops never include a partial inclusion box (all or nothing).
        assert ((include_coverage == 1) | (include_coverage == 0)).all()
        # All inclusion boxes are covered.
        assert include_coverage.max(axis=1).sum() == len(include)


def test_serialization():
    scene_i = mds.load_shapes(n_scenes=1)[0]
    scene_i.metadata = {"foo": "bar"}
    scene_i.annotations[0].metadata = {"baz": "boo"}
    scene_i.masks = [
        {
            "visible": False,
            "name": "test",
            "contour": np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
        }
    ]
    scene_o = mc.Scene.fromString(scene_i.toString())
    np.testing.assert_equal(scene_o.image, scene_i.image)
    np.testing.assert_equal(scene_o.annotated(), scene_i.annotated())
    assert scene_o.metadata == scene_i.metadata
    assert scene_o.annotations == scene_i.annotations


def test_safe_crop():
    size = 8
    config = mc.Categories(["square"])
    canvas = np.zeros((256, 256, 3), dtype="uint8")
    annotations = []
    for x, y in itertools.permutations(np.arange(32, 256, 64), 2):
        annotations.append(
            mc.Annotation(
                x1=x - size,
                y1=y - size,
                x2=x + size,
                y2=y + size,
                category=config["square"],
            )
        )
        cv2.rectangle(
            canvas,
            pt1=(x - size, y - size),
            pt2=(x + size, y + size),
            color=(0, 0, 255),
            thickness=-1,
        )
    scene = mc.Scene(categories=config, image=canvas, annotations=annotations)
    for wiggle in [True, False]:
        augmenter = mc.augmentations.compose(
            [
                mc.augmentations.RandomCropBBoxSafe(
                    width=size * 4, height=size * 4, prob_box=1.0, wiggle=wiggle
                )
            ]
        )
        positions = []
        for _ in range(25):
            augmented = scene.augment(augmenter)[0]
            assert len(augmented.annotations) == 1
            x1, y1, x2, y2 = augmented.annotations[0].x1y1x2y2()
            positions.append([x1, y1, x2, y2])
            image = augmented.image.copy()
            assert (image[y1 : y2 + 1, x1 : x2 + 1, -1] == 255).all()
            image[y1 : y2 + 1, x1 : x2 + 1, -1] = 0
            assert (image == 0).all()
        variety = np.unique(np.array(positions)[:, 0]).shape[0]
        if wiggle:
            assert variety > 5
        else:
            assert variety == 1
        # Make sure it works on empty images.
        scene.assign(annotations=[]).augment(augmenter)


def test_split_apply_combine():
    def process(arr):
        print("Length", len(arr))
        verify = [i % 2 == 0 for i in arr]
        assert all(verify) or not any(verify)
        if verify[0]:
            return arr
        return [i - 1 for i in arr]

    processed = mc.utils.split_apply_combine(
        list(range(10)), key=lambda x: str(x % 2 == 0), func=process
    )
    assert all(e == a for e, a in zip([0, 0, 2, 2, 4, 4, 6, 6, 8, 8], processed))
