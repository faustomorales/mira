"""
Test core functions.
"""
import typing
import itertools
import collections
import tempfile
import os
import io

import numpy as np

from mira import core
from mira.core import experimental as mce
import mira.datasets as mds


def test_blank_and_properties():
    """Make sure creating new images works."""
    image = core.utils.get_blank_image(width=200, height=100, n_channels=2, cval=125)
    assert image.shape == (100, 200, 2)
    assert (image[:, :, :] == 125).all()


def test_scene_deduplication():
    ac = core.AnnotationConfiguration(["foo", "bar"])
    scene = core.Scene(
        image=core.utils.get_blank_image(width=200, height=100, n_channels=2, cval=125),
        annotations=[
            core.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                metadata={"tag": "drop-me-coverage"},
            ),
            core.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=20,
                y2=20,
                metadata={"tag": "drop-me-always"},
            ),
            core.Annotation(
                category=ac["foo"],
                x1=0,
                y1=0,
                x2=20,
                y2=20,
                metadata={"tag": "keep-me"},
            ),
            core.Annotation(
                category=ac["bar"],
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                metadata={"tag": "keep-me"},
            ),
        ],
        annotation_config=ac,
    )
    deduplicated_coverage = scene.drop_duplicates(method="coverage")
    deduplicated_iou = scene.drop_duplicates(method="iou")
    assert len(deduplicated_coverage.annotations) == 2
    assert len(deduplicated_iou.annotations) == 3
    assert all(
        ann.metadata["tag"] == "keep-me" for ann in deduplicated_coverage.annotations
    )
    assert all(
        ann.metadata["tag"] in ["keep-me", "drop-me-coverage"]
        for ann in deduplicated_iou.annotations
    )


def test_file_read():
    """Make sure reading files works."""
    image = core.utils.get_blank_image(width=200, height=100, n_channels=3, cval=125)
    image[40:60, 40:60, 0] = 0
    with tempfile.TemporaryDirectory() as tempdir:
        fpath = os.path.join(tempdir, "test.png")
        core.utils.save(image, fpath)
        np.testing.assert_allclose(core.utils.read(fpath), image)
        with open(fpath, "rb") as buffer:
            np.testing.assert_allclose(core.utils.read(buffer), image)
        with io.BytesIO() as buffer:
            core.utils.save(image, buffer, ".png")
            buffer.seek(0)
            np.testing.assert_allclose(core.utils.read(buffer), image)


def test_split():
    n = 5000
    items = np.arange(n)
    sizes = [0.5, 0.30, 0.20]
    group: typing.List[int] = np.random.choice(500, size=n).tolist()
    stratify: typing.List[int] = np.random.choice(2, size=n).tolist()

    splits1A = core.utils.split(
        items=items, sizes=sizes, stratify=stratify, group=group, random_state=42
    )

    splits1B = core.utils.split(
        items=items, sizes=sizes, stratify=stratify, group=group, random_state=42
    )

    splits2 = core.utils.split(
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
    )

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
            or core.utils.compute_iou(include, exclude).max() > 0
        ):
            continue
        crops = mce.find_acceptable_crops(
            include=include, width=width, height=height, exclude=exclude
        )
        include_coverage = core.utils.compute_coverage(include, crops)
        exclude_coverage = core.utils.compute_coverage(exclude, crops)
        # Crops never include an exclusion box.
        assert (exclude_coverage == 0).all()
        # Crops never include a partial inclusion box (all or nothing).
        assert ((include_coverage == 1) | (include_coverage == 0)).all()
        # All inclusion boxes are covered.
        assert include_coverage.max(axis=1).sum() == len(include)
