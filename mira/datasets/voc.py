"""VOC parsing tools"""

import os
import typing
from xml.etree import ElementTree
import logging

from tqdm import tqdm

from .. import core

log = logging.getLogger(__name__)  # pylint: disable=invalid-name

VOC_SCENE_METADATA_MAP = [
    ["folder"],
    ["filename"],
    ["size", "width"],
    ["size", "height"],
    ["source", "database"],
    ["source", "annotation"],
    ["source", "image"],
    ["segmented"],
]

VOC_ANNOTATION_METADATA_MAP = [["pose"], ["truncated"], ["difficult"]]


def map_xml_to_metadata(
    paths: typing.List[typing.List[str]], root: ElementTree.Element
):
    """Map XML paths into metadata."""
    metadata = {}
    for path in paths:
        key = ":".join(path)
        elem: typing.Optional[ElementTree.Element] = root
        for part in path:
            assert elem is not None
            elem = elem.find(part)
            if elem is None:
                log.info("Missing annotation metadata: %s", ":".join(path))
                break
        if elem is None:
            continue
        metadata[key] = elem.text
    return metadata


def load_voc(
    filepaths: typing.List[str],
    categories: core.Categories,
    image_dir: str = None,
) -> core.SceneCollection:
    """Read a scene from a VOC XML annotation file. Remaining arguments
    passed to scene constructor.

    Args:
        filepaths: A list of VOC files to read
        image_folder: Folder in which to look for images. Defaults to same
            folder as XML file prepended to the folder specified in the
            XML file.
        categories: The annotation configuration to use.

    Returns:
        A new scene collection, one scene per VOC file
    """
    scenes = []
    for filepath in tqdm(filepaths, desc="Loading VOC annotation files."):
        annotations = []
        root = ElementTree.parse(filepath).getroot()

        # Get the scene level metadata
        scene_metadata = map_xml_to_metadata(paths=VOC_SCENE_METADATA_MAP, root=root)

        if image_dir is None:
            folder = scene_metadata["folder"]
            if folder is None:
                image_dir = os.path.dirname(filepath)
            else:
                image_dir = os.path.join(os.path.dirname(filepath), folder)

        image_path = os.path.join(image_dir, scene_metadata["filename"])
        for obj in root.findall("object"):
            category = categories[obj.find("name").text]  # type: ignore
            bndbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = [
                int(float(bndbox.find(k).text))  # type: ignore
                for k in ["xmin", "ymin", "xmax", "ymax"]
            ]
            annotations.append(
                core.Annotation(x1=xmin, y1=ymin, x2=xmax, y2=ymax, category=category)
            )
        scenes.append(
            core.Scene(
                categories=categories,
                annotations=annotations,
                image=image_path,
            )
        )
    return core.SceneCollection(scenes=scenes, categories=categories)
