"""VOC parsing tools"""

import os
from typing import List
from xml.etree import ElementTree
import logging

from tqdm import tqdm

from ..core import (Selection, Scene, SceneCollection, AnnotationConfiguration,
                    Annotation)

log = logging.getLogger(__name__)  # pylint: disable=invalid-name

VOC_SCENE_METADATA_MAP = [['folder'], ['filename'], ['size', 'width'],
                          ['size', 'height'], ['source', 'database'],
                          ['source', 'annotation'], ['source', 'image'],
                          ['segmented']]

VOC_ANNOTATION_METADATA_MAP = [['pose'], ['truncated'], ['difficult']]


def map_xml_to_metadata(paths: List[List[str]], root: ElementTree.Element):
    metadata = {}
    for path in paths:
        key = ':'.join(path)
        elem = root
        for part in path:
            elem = elem.find(part)
            if elem is None:
                log.info('Missing annotation metadata: %s', ':'.join(path))
                break
        if elem is None:
            continue
        metadata[key] = elem.text
    return metadata


def load_voc(
        filepaths: str,
        annotation_config: AnnotationConfiguration,
        image_dir: str = None,
) -> SceneCollection:
    """Read a scene from a VOC XML annotation file. Remaining arguments
    passed to scene constructor.

    Args:
        filepaths: A list of VOC files to read
        image_folder: Folder in which to look for images. Defaults to same
            folder as XML file prepended to the folder specified in the
            XML file.
        annotation_config: The annotation configuration to use.

    Returns:
        A new scene collection, one scene per VOC file
    """
    scenes = []
    for filepath in tqdm(filepaths, desc='Loading VOC annotation files.'):
        annotations = []
        root = ElementTree.parse(filepath).getroot()

        # Get the scene level metadata
        scene_metadata = map_xml_to_metadata(
            paths=VOC_SCENE_METADATA_MAP, root=root)

        if image_dir is None:
            folder = scene_metadata['folder']
            if folder is None:
                image_dir = os.path.dirname(filepath)
            else:
                image_dir = os.path.join(os.path.dirname(filepath), folder)

        image_path = os.path.join(image_dir, scene_metadata['filename'])
        for obj in root.findall('object'):
            category = annotation_config[obj.find('name').text]
            selection = None
            for bndbox in obj.findall('bndbox'):
                xmin, ymin, xmax, ymax = [
                    int(float(bndbox.find(k).text))
                    for k in ['xmin', 'ymin', 'xmax', 'ymax']
                ]
                current = Selection([[xmin, ymin], [xmax, ymax]])
                if selection is None:
                    selection = current
                else:
                    selection += current
            annotations.append(
                Annotation(selection=selection, category=category))
        scenes.append(
            Scene(
                annotation_config=annotation_config,
                annotations=annotations,
                image=image_path))
    return SceneCollection(scenes=scenes, annotation_config=annotation_config)
