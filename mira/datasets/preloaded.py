"""Datasets available out of the box."""
# pylint: disable=invalid-name,line-too-long

import logging
from os import path
from itertools import product

from glob import glob
import numpy as np
import cv2
from pkg_resources import resource_string

from .. import utils
from .. import core
from .voc import load_voc

log = logging.getLogger(__name__)

COCOAnnotationConfiguration = core.AnnotationConfiguration(
    resource_string(__name__, "assets/coco_classes.txt").decode("utf-8").split("\n")
)
VOCAnnotationConfiguration = core.AnnotationConfiguration(
    resource_string(__name__, "assets/voc_classes.txt").decode("utf-8").split("\n")
)
COCOAnnotationConfiguration90 = core.AnnotationConfiguration(
    resource_string(__name__, "assets/coco_classes_90.txt").decode("utf-8").split("\n")
)


def load_random_images():
    """Get some random images from the internet."""
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b7/Atlantic_Puffin.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/1/11/Freightliner_M2_106_6x4_2014_%2814240376744%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/2c/1996_Porsche_911_993_GT2_-_Flickr_-_The_Car_Spy_%284%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/1/1e/Handheld_blowdryer.jpg",
    ]
    annotation_config = core.AnnotationConfiguration([])
    return core.SceneCollection(
        annotation_config=annotation_config,
        scenes=[
            core.Scene(
                image=core.utils.read(url),
                annotations=[],
                annotation_config=annotation_config,
            )
            for url in urls
        ],
    )


def load_voc2012(subset="train") -> core.SceneCollection:
    """PASCAL VOC 2012

    Args:
        subset: One of `train`, `val`, or `trainval`.
            If `trainval`, the scene collection contains both the
            train and validation sets. If `train`, only the
            training set. If `val` only the validation
            set.

    Returns:
        A scene collection containing the PASCAL VOC 2012
        dataset.
    """
    assert subset in [
        "train",
        "val",
        "trainval",
    ], "Subset must be one of train, val, or trainval"

    def extract_check_fn(root_dir):
        image_dir = path.join(root_dir, "vocdevkit", "voc2012", "jpegimages")
        image_files = glob(path.join(image_dir, "*.jpg"))
        annotation_dir = path.join(root_dir, "vocdevkit", "voc2012", "annotations")
        annotation_files = glob(path.join(annotation_dir, "*.xml"))
        return len(annotation_files) == 17125 and len(image_files) == 17125

    root_dir = utils.get_file(
        origin="http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        file_hash="e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb",
        cache_subdir=path.join("datasets", "voc2012"),
        hash_algorithm="sha256",
        extract=True,
        archive_format="tar",
        extract_check_fn=extract_check_fn,
    )
    image_dir = path.join(root_dir, "vocdevkit", "voc2012", "jpegimages")
    annotation_dir = path.join(root_dir, "vocdevkit", "voc2012", "annotations")
    imageset_dir = path.join(root_dir, "vocdevkit", "voc2012", "imagesets", "main")
    sids = []
    if subset in ["trainval", "train"]:
        sid_path = path.join(imageset_dir, "train.txt")
        with open(sid_path, "r", encoding="utf8") as f:
            sids.extend(f.read().split("\n"))
    if subset in ["trainval", "val"]:
        sid_path = path.join(imageset_dir, "val.txt")
        with open(sid_path, "r", encoding="utf8") as f:
            sids.extend(f.read().split("\n"))
    filepaths = [
        path.join(annotation_dir, f"{sid}.xml") for sid in sids if len(sid.strip()) > 0
    ]
    return load_voc(
        filepaths=filepaths,
        annotation_config=VOCAnnotationConfiguration,
        image_dir=image_dir,
    )


def load_shapes(
    width=256,
    height=256,
    object_count_bounds=(3, 8),
    object_width_bounds=(20, 40),
    n_scenes=100,
    polygons=False,
) -> core.SceneCollection:
    """A simple dataset for testing.

    Args:
        width: The width of each image
        height: The height of each image
        object_count_bounds: A tuple indicating the minimum and maximum
            number of objects in each image
        object_width_bounds: A tuple indicating the minimum and maximum
            widths of the objects in each image
        n_scenes: The number of scenes to generate
        polygons: Whether to use polygons instead of axis-aligned
            bounding boxes.

    Returns:
        A scene collection of images with circles and rectangles in it.
    """
    annotation_config = core.AnnotationConfiguration(
        [
            " ".join([s, c])
            for s, c in product(["RED", "BLUE", "GREEN"], ["RECTANGLE", "CIRCLE"])
        ]
    )

    def make_scene():
        # pylint: disable=unsubscriptable-object
        image = core.utils.get_blank_image(
            width=width, height=height, n_channels=3, cval=255
        )
        object_count = np.random.randint(*object_count_bounds)
        ws = np.random.randint(object_width_bounds[0], object_width_bounds[1], size=object_count)  # type: ignore
        xs = np.random.randint(
            low=0, high=width - object_width_bounds[-1], size=object_count
        )
        ys = np.random.randint(
            low=0, high=height - object_width_bounds[-1], size=object_count
        )
        shapes = np.random.choice(["RECTANGLE", "CIRCLE"], size=object_count)
        colors = np.random.choice(["RED", "BLUE", "GREEN"], size=object_count)
        lookup = {"RED": (255, 0, 0), "BLUE": (0, 0, 255), "GREEN": (0, 255, 0)}
        annotations = []
        for x, y, w, shape, color in zip(xs, ys, ws, shapes, colors):
            if image[y : y + w, x : x + w].min() == 0:
                # Avoid overlapping shapes.
                continue
            if shape == "RECTANGLE":
                cv2.rectangle(
                    image,
                    pt1=(x, y),
                    pt2=(x + w, y + w),
                    thickness=-1,
                    color=lookup[color],
                )
                points = [(x, y), (x + w, y), (x + w, y + w), (x, y + w), (x, y)]
            elif shape == "CIRCLE":
                r = w // 2
                w = 2 * r
                cv2.circle(
                    image,
                    center=(x + r, y + r),
                    radius=r,
                    thickness=-1,
                    color=lookup[color],
                )
                t = np.linspace(0, 2 * np.pi, num=20)
                points = np.array(
                    [x + r * (np.cos(t) + 1), y + r * (1 + np.sin(t))]
                ).T.tolist()
            if polygons:
                annotation = core.Annotation(
                    points=points,
                    category=annotation_config[" ".join([color, shape])],
                )
            else:
                annotation = core.Annotation(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + w,
                    category=annotation_config[" ".join([color, shape])],
                )
            annotations.append(annotation)
        return core.Scene(
            annotations=annotations, image=image, annotation_config=annotation_config
        )

    scenes = [make_scene() for n in range(n_scenes)]
    return core.SceneCollection(scenes=scenes, annotation_config=annotation_config)


def load_oxfordiiitpets(breed=True) -> core.SceneCollection:
    """Load the Oxford-IIIT pets dataset. It is not divided into
    train, validation, and test because it appeared some files were missing
    from the trainval and test set documents
    (e.g., english_cocker_spaniel_164).

    Args:
        breed: Whether to use the breeds as the class labels. If False, the
            class labels are limited to dog or cat.

    Returns:
        A scene collection containing the dataset
    """
    image_dir = utils.get_file(
        origin="http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        file_hash="67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
        cache_subdir=path.join("datasets", "oxfordiiitpets"),
        hash_algorithm="sha256",
        extract=True,
        archive_format="tar",
        extract_check_fn=lambda directory: len(
            glob(path.join(directory, "images", "*.jpg"))
        )
        == 7390,
    )
    annotations_dir = utils.get_file(
        origin="http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        file_hash="52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
        cache_subdir=path.join("datasets", "oxfordiiitpets"),
        hash_algorithm="sha256",
        extract=True,
        archive_format="tar",
        extract_check_fn=lambda directory: len(
            glob(path.join(directory, "annotations", "xmls", "*.xml"))
        )
        == 3686,
    )
    filepaths = glob(path.join(annotations_dir, "annotations", "xmls", "*.xml"))
    image_dir = path.join(image_dir, "images")
    collection = load_voc(
        filepaths=filepaths,
        annotation_config=core.AnnotationConfiguration(["dog", "cat"]),
        image_dir=image_dir,
    )
    if not breed:
        return collection
    assert all(
        len(s.annotations) in [1, 2] for s in collection.scenes
    ), "An error occurred handling pets dataset"
    labels = [
        "_".join(path.splitext(path.split(f)[1])[0].split("_")[:-1]) for f in filepaths
    ]
    annotation_config = core.AnnotationConfiguration(sorted(set(labels)))
    return core.SceneCollection(
        scenes=[
            scene.assign(
                annotations=[
                    a.assign(category=annotation_config[label])
                    for a in scene.annotations
                ],
                annotation_config=annotation_config,
            )
            for scene, label in zip(collection.scenes, labels)
        ],
        annotation_config=annotation_config,
    )
