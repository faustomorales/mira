"""COCO parsing tools"""
# pylint: disable=invalid-name

import logging
from os import path
import json

from tqdm import tqdm
import numpy as np

from ..core import (
    Scene,
    SceneCollection,
    AnnotationConfiguration,
    Annotation,
)

log = logging.getLogger(__name__)


def load_coco(
    annotations_file: str,
    image_dir: str,
    annotation_config: AnnotationConfiguration = None,
) -> SceneCollection:
    """Obtain a scene collection from a COCO JSON file.

    Args:
        annotations_file: The annotation file to load
        image_dir: The directory in which to look for images
        annotation_config: The annotation configuration to
            use. If None, it is inferred from the annotations
            category file.

    Returns:
        A scene collection
    """
    if image_dir is None:
        image_dir = path.split(annotations_file)[0]
    with open(annotations_file, "r", encoding="utf8") as f:
        data = json.load(f)
    categories = {}
    for category in data["categories"]:
        categories[category["id"]] = category["name"]
    category_names = [
        c[1] for c in sorted(list(categories.items()), key=lambda x: x[0])
    ]
    if annotation_config is None:
        annotation_config = AnnotationConfiguration(category_names)
    assert len(categories) == len(
        annotation_config
    ), "Annotation configuration incompatible with in-file categories"
    assert all(
        c in annotation_config for c in category_names
    ), "Some in-file categories not in annotation configuration"
    assert all(
        c.name in category_names for c in annotation_config
    ), "Some annotation configuration categories not in file"

    annotations = np.array(
        [
            [ann["image_id"], ann["category_id"]] + ann["bbox"]
            for ann in data["annotations"]
        ]
    )
    annotations = annotations[annotations[:, 0].argsort()]
    images = sorted(data["images"], key=lambda x: x["id"])
    del data
    scenes = []
    startIdx = 0
    for image in tqdm(images, total=len(images), desc="Creating scenes"):
        current = annotations[startIdx:][annotations[startIdx:, 0] == image["id"], 1:]
        startIdx += len(current)
        scenes.append(
            Scene(
                image=path.join(image_dir, image["file_name"]),
                annotation_config=annotation_config,
                annotations=[
                    Annotation(
                        category=annotation_config[categories[int(ann[0])]],
                        x1=ann[1],
                        y1=ann[2],
                        x2=ann[1] + ann[3],
                        y2=ann[2] + ann[4],
                    )
                    for ann in current
                ],
            )
        )
    return SceneCollection(scenes=scenes, annotation_config=annotation_config)


def load_coco_text(
    annotations_file: str,
    image_dir: str,
    annotation_config: AnnotationConfiguration = None,
) -> SceneCollection:
    """Obtain a scene collection from a COCO Text JSON file
    (e.g., that which can be obtained from https://bgshih.github.io/cocotext/)

    Args:
        annotations_file: The annotation file to load
        image_dir: The directory in which to look for images
        annotation_config: The annotation configuration to
            use. If None, it is inferred from the annotations
            category file.

    Returns:
        A scene collection
    """
    with open(annotations_file, "r", encoding="utf8") as f:
        data = json.load(f)

    category_names = set(ann["class"] for ann in data["anns"].values())
    if annotation_config is None:
        annotation_config = AnnotationConfiguration(sorted(list(category_names)))
    assert len(category_names) == len(
        annotation_config
    ), "Annotation configuration incompatible with in-file categories"
    assert all(
        c in annotation_config for c in category_names
    ), "Some in-file categories not in annotation configuration"
    assert all(
        c.name in category_names for c in annotation_config
    ), "Some annotation configuration categories not in file"

    images = data["imgs"]
    scenes = []
    for (imageId, imageData) in tqdm(
        images.items(), total=len(images), desc="Creating scenes"
    ):
        anns = [data["anns"][str(annId)] for annId in data["imgToAnns"][imageId]]
        scenes.append(
            Scene(
                annotation_config=annotation_config,
                image=path.join(image_dir, imageData["file_name"]),
                annotations=[
                    Annotation(
                        category=annotation_config[ann["class"]],
                        x1=ann["bbox"][0],
                        y1=ann["bbox"][1],
                        x2=ann["bbox"][0] + ann["bbox"][2],
                        y2=ann["bbox"][1] + ann["bbox"][3],
                    )
                    for ann in anns
                ],
            )
        )
    return SceneCollection(scenes=scenes, annotation_config=annotation_config)
