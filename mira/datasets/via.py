import csv
import json
from os import path
from typing import Union

import validators

from ..core import (
    Annotation,
    Scene,
    SceneCollection,
    Selection,
    AnnotationConfiguration
)


def load_via(
    annotations_file: str,
    label_key: str='class',
    image_dir: str=None,
    annotation_config: AnnotationConfiguration=None,
    cache: Union[str, bool]=True
) -> SceneCollection:
    """Load annotations created by the `VGG Image Annotation Tool
    <http://www.robots.ox.ac.uk/~vgg/software/via/>`_.

    Args:
        annotations_file: The file containing the
            annotations. For now, only the CSV
            format is supported.
        label_key: The attribute name to use for class labels
        annotation_config: The annotation configuration to use.
            If None, it is inferred from the annotations file.
        image_dir: The directory in which to look
            for images. If None, defaults the
            folder in which the annotations_file
            is stored.
        cache: The cache parameter to pass to `Scene`. Useful
            if the annotations file references URLs (i.e.,
            specify a directory in which to save images).
    """
    if not annotations_file.endswith('.csv'):
        raise NotImplementedError(
            'Only the VIA CSV format is supported at '
            'this time.'
        )
    if image_dir is None:
        image_dir = path.split(annotations_file)[0]
    annotations_dict = {}
    category_set = set()
    with open(annotations_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            image_name = row[0]
            if image_name not in annotations_dict:
                annotations_dict[image_name] = []

            region_shape = json.loads(row[5])
            region_attr = json.loads(row[6])

            if len(region_shape) == 0:
                # If there is no region information,
                # it means there is no annotation in this
                # image.
                continue

            if region_shape['name'] == 'rect':
                x = region_shape['x']
                y = region_shape['y']
                width = region_shape['width']
                height = region_shape['height']
                selection = Selection(
                    [
                        [x, y],
                        [x + width, y + height]
                    ]
                )
            else:
                raise NotImplementedError(
                    'Only rectangles are supported at this time, '
                    'not %s' % region_shape['name']
                )
            category_name = region_attr[label_key].strip()
            category_set.update([category_name])
            annotations_dict[image_name].append(
                (category_name, selection)
            )
    if annotation_config is not None:
        assert all(c in annotation_config for c in category_set)
    if annotation_config is None:
        annotation_config = AnnotationConfiguration(
            sorted(list(category_set))
        )
    scenes = []
    for image, annotations in annotations_dict.items():
        if not validators.url(image):
            image = path.join(image_dir, image)
        annotations = [
            Annotation(
                selection=selection,
                category=annotation_config[category_name]
            ) for category_name, selection in annotations
        ]
        scenes.append(
            Scene(
                image=image,
                annotations=annotations,
                annotation_config=annotation_config,
                cache=cache
            )
        )
    return SceneCollection(
        scenes=scenes,
        annotation_config=annotation_config
    )