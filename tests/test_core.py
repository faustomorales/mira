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

import mira.core as mc
import mira.core.augmentations as mca
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


def check_scene_equality(scene1, scene2):
    """Check whether two scenes are equivalent with respect to image, annotations and metadata."""
    np.testing.assert_equal(scene1.image, scene2.image)
    np.testing.assert_equal(scene1.annotated(), scene2.annotated())
    assert scene1.metadata == scene2.metadata
    assert scene1.annotations == scene2.annotations


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
    scene_o = mc.Scene.fromString(*scene_i.toString())
    check_scene_equality(scene_i, scene_o)


def test_collection_serialization():
    collection_i = mds.load_shapes(n_scenes=10)
    with tempfile.TemporaryDirectory() as tdir:
        tarball = os.path.join(tdir, "serialized.tar.gz")
        collection_i.save(tarball)
        for collection_o in [
            mc.SceneCollection.load(tarball, directory=os.path.join(tdir, "extracted")),
            mc.SceneCollection.load(tarball),
        ]:
            for scene_i, scene_o in zip(collection_i, collection_o):
                check_scene_equality(scene_i, scene_o)


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
        augmenter = mca.compose(
            [
                mca.RandomCropBBoxSafe(
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


def test_compute_contour_coverage():
    """Test compute_contour_coverage function."""
    # Create test contours (as rectangles)
    # Contour A1: small box at (0, 0) to (10, 10)
    contourA1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    # Contour A2: box at (50, 50) to (60, 60)
    contourA2 = np.array([[50, 50], [60, 50], [60, 60], [50, 60]])

    # Contour B1: large box containing A1 at (0, 0) to (20, 20)
    contourB1 = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
    # Contour B2: partially overlapping with A2 at (55, 55) to (65, 65)
    contourB2 = np.array([[55, 55], [65, 55], [65, 65], [55, 65]])
    # Contour B3: no overlap with any A contour at (100, 100) to (110, 110)
    contourB3 = np.array([[100, 100], [110, 100], [110, 110], [100, 110]])

    contoursA = [contourA1, contourA2]
    contoursB = [contourB1, contourB2, contourB3]

    coverage = mc.utils.compute_contour_coverage(contoursA, contoursB)

    # Check shape
    assert coverage.shape == (2, 3)

    # A1 should be fully covered by B1 (coverage ~1.0)
    assert coverage[0, 0] > 0.99

    # A1 should have no overlap with B2 or B3
    assert coverage[0, 1] == 0
    assert coverage[0, 2] == 0

    # A2 should have no overlap with B1
    assert coverage[1, 0] == 0

    # A2 should have partial overlap with B2 (less than full coverage)
    assert 0 < coverage[1, 1] < 1.0

    # A2 should have no overlap with B3
    assert coverage[1, 2] == 0


def test_compute_contour_iou():
    """Test compute_contour_iou function."""
    # Create test contours (as rectangles)
    # Contour A1: box at (0, 0) to (10, 10)
    contourA1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    # Contour A2: box at (50, 50) to (60, 60)
    contourA2 = np.array([[50, 50], [60, 50], [60, 60], [50, 60]])

    # Contour B1: identical to A1
    contourB1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    # Contour B2: partially overlapping with A1 at (5, 5) to (15, 15)
    contourB2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15]])
    # Contour B3: no overlap with any A contour at (100, 100) to (110, 110)
    contourB3 = np.array([[100, 100], [110, 100], [110, 110], [100, 110]])

    contoursA = [contourA1, contourA2]
    contoursB = [contourB1, contourB2, contourB3]

    iou = mc.utils.compute_contour_iou(contoursA, contoursB)

    # Check shape
    assert iou.shape == (2, 3)

    # A1 and B1 are identical, so IoU should be ~1.0
    assert iou[0, 0] > 0.99

    # A1 and B2 partially overlap, so IoU should be between 0 and 1
    assert 0 < iou[0, 1] < 1.0

    # A1 should have no overlap with B3
    assert iou[0, 2] == 0

    # A2 should have no overlap with B1 or B2
    assert iou[1, 0] == 0
    assert iou[1, 1] == 0

    # A2 should have no overlap with B3
    assert iou[1, 2] == 0


def test_has_overlap():
    """Test has_overlap function."""
    # Create test boxes in x1y1x2y2 format
    boxesA = np.array([
        [0, 0, 10, 10],      # Box A1
        [50, 50, 60, 60],    # Box A2
        [100, 100, 110, 110] # Box A3
    ])

    boxesB = np.array([
        [5, 5, 15, 15],      # Box B1: overlaps with A1
        [50, 50, 60, 60],    # Box B2: identical to A2
        [200, 200, 210, 210] # Box B3: no overlap with any A box
    ])

    overlap = mc.utils.has_overlap(boxesA, boxesB)

    # Check shape
    assert overlap.shape == (3, 3)

    # Check data type is boolean
    assert overlap.dtype == bool

    # A1 overlaps with B1
    assert overlap[0, 0]

    # A1 does not overlap with B2
    assert not overlap[0, 1]

    # A1 does not overlap with B3
    assert not overlap[0, 2]

    # A2 does not overlap with B1
    assert not overlap[1, 0]

    # A2 overlaps with B2 (identical boxes)
    assert overlap[1, 1]

    # A2 does not overlap with B3
    assert not overlap[1, 2]

    # A3 does not overlap with any B box
    assert not overlap[2, 0]
    assert not overlap[2, 1]
    assert not overlap[2, 2]


def test_annotation_crop():
    """Test crop functionality for both rectangle and polygon annotations."""
    categories = mc.Categories(["test"])
    
    # Test 1: Rectangle annotation fully within crop region
    rect_ann = mc.Annotation(
        x1=10, y1=10, x2=50, y2=50,
        category=categories["test"]
    )
    cropped = rect_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 1
    assert cropped[0].x1 == 10 and cropped[0].y1 == 10
    assert cropped[0].x2 == 50 and cropped[0].y2 == 50
    
    # Test 2: Rectangle annotation partially outside crop region
    rect_ann = mc.Annotation(
        x1=50, y1=50, x2=150, y2=150,
        category=categories["test"]
    )
    cropped = rect_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 1
    assert cropped[0].x1 == 50 and cropped[0].y1 == 50
    assert cropped[0].x2 == 100 and cropped[0].y2 == 100
    
    # Test 3: Rectangle annotation completely outside crop region
    rect_ann = mc.Annotation(
        x1=150, y1=150, x2=200, y2=200,
        category=categories["test"]
    )
    cropped = rect_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 0
    
    # Test 4: Polygon annotation fully within crop region
    # Polygon at (10, 10) to (40, 40), crop region (0, 0) to (100, 100)
    polygon_ann = mc.Annotation(
        points=np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.float64),
        category=categories["test"]
    )
    cropped = polygon_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 1
    assert cropped[0].points.shape[0] >= 4
    # All vertices should be within crop region [0, 0] to [100, 100]
    assert np.all(cropped[0].points >= 0)
    assert np.all(cropped[0].points[:, 0] <= 100)
    assert np.all(cropped[0].points[:, 1] <= 100)
    
    # Test 5: Polygon annotation partially outside crop region
    # Polygon at (50, 50) to (150, 150), crop region (0, 0) to (100, 100)
    # Should be clipped to (50, 50) to (100, 100)
    polygon_ann = mc.Annotation(
        points=np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float64),
        category=categories["test"]
    )
    cropped = polygon_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 1
    assert cropped[0].points.shape[0] >= 4
    # All vertices should be within crop region [0, 0] to [100, 100]
    assert np.all(cropped[0].points >= 0)
    assert np.all(cropped[0].points[:, 0] <= 100)
    assert np.all(cropped[0].points[:, 1] <= 100)
    
    # Test 6: Polygon annotation completely outside crop region
    polygon_ann = mc.Annotation(
        points=np.array([[150, 150], [200, 150], [200, 200], [150, 200]], dtype=np.float64),
        category=categories["test"]
    )
    cropped = polygon_ann.crop(width=100, height=100, xoffset=0, yoffset=0)
    assert len(cropped) == 0
    
    # Test 7: Crop with offset (non-zero xoffset and yoffset)
    # Polygon at (60, 60) to (150, 150), crop region (50, 50) to (150, 150)
    # After crop, coordinates should be relative to (50, 50)
    polygon_ann = mc.Annotation(
        points=np.array([[60, 60], [150, 60], [150, 150], [60, 150]], dtype=np.float64),
        category=categories["test"]
    )
    cropped = polygon_ann.crop(width=100, height=100, xoffset=50, yoffset=50)
    assert len(cropped) == 1
    # Coordinates should be in crop region coordinate space [0, 0] to [100, 100]
    assert np.all(cropped[0].points >= 0)
    assert np.all(cropped[0].points[:, 0] <= 100)
    assert np.all(cropped[0].points[:, 1] <= 100)
