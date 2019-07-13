"""
Tools for loading and saving VGG Image Annotator datasets.
"""

import os
import json
import uuid
import typing

import pandas as pd

from .. import core


def load_via(project_file: str,
             label_key: str = 'class',
             image_dir: str = None,
             cache: typing.Union[str, bool] = True) -> core.SceneCollection:
    """Load annotations created by the `VGG Image Annotation Tool
    <http://www.robots.ox.ac.uk/~vgg/software/via/>`_.

    Args:
        project_file: The file containing the
            annotations. Only the JSON
            format is supported.
        label_key: The attribute name to use for class labels
        image_dir: The directory in which to look
            for images. If None, defaults the
            folder in which the annotations_file
            is stored.
        cache: The cache parameter to pass to `Scene`. Useful
            if the annotations file references URLs (i.e.,
            specify a directory in which to save images).
    """
    if image_dir is None:
        image_dir = os.path.split(project_file)[0]
    if not project_file.endswith('.json'):
        raise NotImplementedError('Only the VIA JSON format is supported.')
    with open(project_file) as f:  # pylint: disable=invalid-name
        project_data = json.loads(f.read())
    img_metadata = project_data['_via_img_metadata'].values()
    project_attrs = project_data['_via_attributes']
    filenames = [img['filename'] for img in img_metadata]
    regions_df = pd.concat([
        pd.io.json.json_normalize(img['regions']).assign(
            filename=img['filename']
        ) for img in img_metadata
    ], axis=0)

    annotation_config = core.AnnotationConfiguration(
        project_attrs['region'][label_key]['options'].keys()
    )
    assert (regions_df['shape_attributes.name'] == 'rect').all(), \
        'Only axis aligned rectangular selections are supported.'
    return core.SceneCollection(
        scenes=[
            core.Scene(
                image=os.path.join(image_dir, filename),
                annotation_config=annotation_config,
                cache=cache,
                annotations=[
                    core.Annotation(
                        selection=core.Selection(
                            points=[
                                [r['shape_attributes.x'], r['shape_attributes.y']],
                                [
                                    r['shape_attributes.x'] + r['shape_attributes.width'],
                                    r['shape_attributes.y'] + r['shape_attributes.height']
                                ]
                            ]
                        ),
                        category=annotation_config[r[f'region_attributes.{label_key}']]
                    )
                    for _, r in regions_df[regions_df.filename == filename].iterrows()
                ]
            ) for filename in filenames
        ],
        annotation_config=annotation_config
    )

def save_via(
        collection: core.SceneCollection,
        export_dir: str,
        label_key='class',
        project_name='mira_export'
):
    """Save a scene collection in VIA format.

    Args:
        collection: The scene collection to save.
        export_dir: The directory where the via.json file along with all images
            will be saved.
        label_key: The region attribute in which to store the labels
    """
    # Make sure the directory does not exist.
    if os.path.isdir(export_dir):
        raise ValueError(f'{export_dir} already exists.')
    os.makedirs(export_dir, exist_ok=False)
    data = [(f'{uuid.uuid4()}.jpg', scene) for scene in collection]
    for filename, scene in data:
        scene.image.save(os.path.join(export_dir, filename))
    data = [
        (filename, scene, os.path.getsize(os.path.join(export_dir, filename)))
        for filename, scene in data
    ]
    sc_bboxes = [[ann.selection.xywh() for ann in s.annotations] for s in collection]
    img_metadata_keys = [f'{filename}{size}' for filename, _, size in data]
    img_metadata_values = [
        {
            'filename': filename,
            'size': size,
            'regions': [
                {
                    'shape_attributes': {
                        'name': 'rect',
                        'x': x, 'y': y, 'width': width, 'height': height
                    },
                    'region_attributes': {
                        label_key: ann.category.name
                    }
                } for (x, y, width, height), ann in zip(bboxes, scene.annotations)
            ]
        } for (filename, scene, size), bboxes in zip(data, sc_bboxes)
    ]
    via_attributes = {
        'region': {
            label_key: {
                'type': 'dropdown',
                'description': '',
                'options': dict(
                    zip(
                        [category.name for category in collection.annotation_config],
                        ["" for category in collection.annotation_config]
                    )
                ),
                'default_options': {}
            }
        },
        'file': {}
    }
    export_object = {
        '_via_img_metadata': dict(zip(img_metadata_keys, img_metadata_values)),
        '_via_attributes': via_attributes,
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all"
                },
                "image": {
                    "region_label": "__via_region_id__",
                    "region_color": "__via_default_region_color__",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION"
                }
            },
            "core": {
                "buffer_size": 18,
                "filepath": {},
                "default_filepath": ""
            },
            "project": {
                "name": project_name
            }
        }
    }
    with open(os.path.join(export_dir, 'via.json'), 'w') as f:  # pylint: disable=invalid-name
        f.write(json.dumps(export_object))
