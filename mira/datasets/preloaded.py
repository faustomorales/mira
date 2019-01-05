import logging
from os import path
from itertools import product

from glob import glob
import numpy as np
import cv2
import re

from .. import utils
from ..core import (
    Image,
    Scene,
    Selection,
    SceneCollection,
    Annotation,
    AnnotationConfiguration
)
from .voc import load_voc


log = logging.getLogger(__name__)


def load_voc2012(subset='train') -> SceneCollection:
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
    assert subset in ['train', 'val', 'trainval'], \
        'Subset must be one of train, val, or trainval'

    def extract_check_fn(root_dir):
        image_dir = path.join(root_dir, 'vocdevkit', 'voc2012', 'jpegimages')
        image_files = glob(path.join(image_dir, '*.jpg'))
        annotation_dir = path.join(root_dir, 'vocdevkit', 'voc2012', 'annotations')  # noqa: E501
        annotation_files = glob(path.join(annotation_dir, '*.xml'))
        return (
            len(annotation_files) == 17125 and
            len(image_files) == 17125
        )
    root_dir = utils.get_file(
        origin='http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',  # noqa: E501
        file_hash='e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb',  # noqa: E501
        cache_subdir=path.join('datasets', 'voc2012'),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        extract_check_fn=extract_check_fn
    )
    image_dir = path.join(root_dir, 'vocdevkit', 'voc2012', 'jpegimages')
    annotation_dir = path.join(root_dir, 'vocdevkit', 'voc2012', 'annotations')
    imageset_dir = path.join(root_dir, 'vocdevkit', 'voc2012', 'imagesets', 'main')  # noqa: E501
    sids = []
    if subset in ['trainval', 'train']:
        sid_path = path.join(imageset_dir, 'train.txt')
        with open(sid_path, 'r') as f:
            sids.extend(f.read().split('\n'))
    if subset in ['trainval', 'val']:
        sid_path = path.join(imageset_dir, 'val.txt')
        with open(sid_path, 'r') as f:
            sids.extend(f.read().split('\n'))
    filepaths = [
        path.join(annotation_dir, '{0}.xml'.format(sid))
        for sid in sids if len(sid.strip()) > 0
    ]
    return load_voc(
        filepaths=filepaths,
        annotation_config=AnnotationConfiguration.VOC,
        image_dir=image_dir
    )


def load_shapes(
    width=256,
    height=256,
    object_count_bounds=(3, 8),
    object_width_bounds=(20, 40),
    n_scenes=100,
) -> SceneCollection:
    """A simple dataset for testing.

    Args:
        width: The width of each image
        height: The height of each image
        object_count_bounds: A tuple indicating the minimum and maximum
            number of objects in each image
        object_width_bounds: A tuple indicating the minimum and maximum
            widths of the objects in each image
        n_scenes: The number of scenes to generate

    Returns:
        A scene collection of images with circles and rectangles in it.
    """
    annotation_config = AnnotationConfiguration(
        [
            ' '.join([s, c]) for s, c in product(
                ['RED', 'BLUE', 'GREEN'],
                ['RECTANGLE', 'CIRCLE']
            )
        ]
    )

    def make_scene():
        image = Image.new(width=width, height=height, channels=3, cval=255)
        object_count = np.random.randint(*object_count_bounds)
        ws = np.random.randint(*object_width_bounds, size=object_count)
        xs = np.random.randint(
            low=0,
            high=width - object_width_bounds[-1],
            size=object_count
        )
        ys = np.random.randint(
            low=0,
            high=height - object_width_bounds[-1],
            size=object_count
        )
        shapes = np.random.choice(['RECTANGLE', 'CIRCLE'], size=object_count)
        colors = np.random.choice(['RED', 'BLUE', 'GREEN'], size=object_count)
        lookup = {
            'RED': (255, 0, 0),
            'BLUE': (0, 0, 255),
            'GREEN': (0, 255, 0)
        }
        annotations = []
        for x, y, w, shape, color in zip(xs, ys, ws, shapes, colors):
            if shape == 'RECTANGLE':
                cv2.rectangle(
                    image,
                    pt1=(x, y),
                    pt2=(x+w, y+w),
                    thickness=-1,
                    color=lookup[color]
                )
            elif shape == 'CIRCLE':
                r = w // 2
                w = 2*r
                cv2.circle(
                    image,
                    center=(x+r, y+r),
                    radius=r,
                    thickness=-1,
                    color=lookup[color]
                )
            annotations.append(Annotation(
                selection=Selection([
                    [x, y],
                    [x + w, y + w]
                ]),
                category=annotation_config[' '.join([color, shape])]
            ))
        return Scene(
            annotations=annotations,
            image=image,
            annotation_config=annotation_config
        )
    scenes = [make_scene() for n in range(n_scenes)]
    return SceneCollection(
        scenes=scenes,
        annotation_config=annotation_config
    )


def load_oxfordiiitpets(breed=True) -> SceneCollection:
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
        origin='http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',  # noqa: E501
        file_hash='67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d',  # noqa: E501
        cache_subdir=path.join('datasets', 'oxfordiiitpets'),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        extract_check_fn=lambda directory: len(glob(path.join(directory, 'images', '*.jpg'))) == 7390  # noqa: E501
    )
    annotations_dir = utils.get_file(
        origin='http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz',  # noqa: E501
        file_hash='52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91',  # noqa: E501
        cache_subdir=path.join('datasets', 'oxfordiiitpets'),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        extract_check_fn=lambda directory: len(glob(path.join(directory, 'annotations', 'xmls', '*.xml'))) == 3686  # noqa: E501
    )
    filepaths = glob(
        path.join(annotations_dir, 'annotations', 'xmls', '*.xml')
    )
    image_dir = path.join(image_dir, 'images')
    collection = load_voc(
        filepaths=filepaths,
        annotation_config=AnnotationConfiguration(['dog', 'cat']),
        image_dir=image_dir
    )
    if not breed:
        return collection
    assert all(len(s.annotations) in [1, 2] for s in collection.scenes), \
        'An error occurred handling pets dataset'
    labels = ['_'.join(path.splitext(path.split(f)[1])[0].split('_')[:-1]) for f in filepaths]  # noqa: E501
    annotation_config = AnnotationConfiguration(sorted(set(labels)))
    return SceneCollection(
        scenes=[
            scene.assign(
                annotations=[a.assign(category=annotation_config[label]) for a in scene.annotations],  # noqa: E501
                annotation_config=annotation_config
            ) for scene, label in zip(collection.scenes, labels)
        ],
        annotation_config=annotation_config
    )


def load_icdar2015(
    subset: str='train',
    text_category: str='text'
) -> SceneCollection:
    """Loads dataset from 2015 Robust Reading Competition.
    More details available at http://rrc.cvc.uab.es/?ch=4&com=introduction

    Args:
        subset: One of `train`, `test`, or `traintest`.
            If `traintest`, the scene collection contains both the
            train and test sets. If `train`, only the
            training set. If `test` only the test set.
        text_category: The category name to use for the
            annotation configuration.

    Returns:
        A scene collection containing the ICDAR 2015 dataset
        dataset.
    """

    scenes = []
    annotation_config = AnnotationConfiguration([text_category])

    def annotation_from_box(box):
        # We have to do this because there are no quotes
        # around text that contains commas in the annotation files.
        # For an example, see gt_img_115.txt.
        elements = box.split(',')
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, elements[:8])
        text = ','.join(elements[8:])  # noqa: F841
        return Annotation(
            selection=Selection(np.array([
                [x1, y1],
                [x2, y2],
                [x3, y3],
                [x4, y4]
            ])),
            category=annotation_config[text_category]
        )

    def scene_from_files(annotation_file, image_file):
        with open(annotation_file, 'r', encoding='utf-8-sig') as f:
            annotations = [
                annotation_from_box(box) for box in f.read().split('\n')[:-1]
            ]
        return Scene(
            image=image_file,
            annotations=annotations,
            annotation_config=annotation_config
        )

    def scenes_from_directories(annotations_dir, images_dir):
        pattern = re.compile(r'img_([0-9]+).')
        image_files = sorted(
            glob(path.join(images_dir, '*.jpg')),
            key=lambda x: pattern.findall(x)[0]
        )
        annotation_files = sorted(
            glob(path.join(annotations_dir, '*.txt')),
            key=lambda x: pattern.findall(x)[0]
        )
        assert len(image_files) == len(annotation_files), \
            'An error occurred loading the dataset.'
        p = utils.Progbar(
            len(annotation_files),
            task_name='Creating scenes from ' + annotations_dir
        )
        scenes = []
        for i, (annotation_file, image_file) in enumerate(
            zip(annotation_files, image_files)
        ):
            p.update(i + 1)
            scenes.append(
                scene_from_files(annotation_file, image_file)
            )
        return scenes

    if subset in ['train', 'traintest']:
        images_dir = utils.get_file(
            origin='https://storage.googleapis.com/miradata/datasets/rrc2015/ch4_training_images.zip',  # noqa: E501
            file_hash='07e2d816dab67df4cd509641b1e1e8c720d3b15873a65b6dc9c1aa6a4de3d2c6',  # noqa: E501
            cache_subdir=path.join('datasets', 'rrc2015'),
            hash_algorithm='sha256',
            extract=True,
            archive_format='zip',
            extract_check_fn=lambda directory: len(glob(path.join(directory, '*.jpg'))) == 1000  # noqa: E501
        )
        annotations_dir = utils.get_file(
            origin='https://storage.googleapis.com/miradata/datasets/rrc2015/ch4_training_localization_transcription_gt.zip',  # noqa: E501
            file_hash='b9f2f0343d016a326bcafe3c28e0ddbda18b3a1e4a8a595784ba9dc3e305d754',  # noqa: E501
            cache_subdir=path.join('datasets', 'rrc2015'),
            hash_algorithm='sha256',
            extract=True,
            archive_format='zip',
            extract_check_fn=lambda directory: len(glob(path.join(directory, '*.txt'))) == 1000  # noqa: E501
        )
        scenes += scenes_from_directories(
            annotations_dir,
            images_dir
        )
    if subset in ['test', 'traintest']:
        images_dir = utils.get_file(
            origin='https://storage.googleapis.com/miradata/datasets/rrc2015/ch4_test_images.zip',  # noqa: E501
            file_hash='ecfb2488333372b7381b14ccb5ac2127de88fb935f6759fe89e5c041b0e87358',  # noqa: E501
            cache_subdir=path.join('datasets', 'rrc2015'),
            hash_algorithm='sha256',
            extract=True,
            archive_format='zip',
            extract_check_fn=lambda directory: len(glob(path.join(directory, '*.jpg'))) == 500  # noqa: E501
        )
        annotations_dir = utils.get_file(
            origin='https://storage.googleapis.com/miradata/datasets/rrc2015/Challenge4_Test_Task1_GT.zip',  # noqa: E501
            file_hash='8f434c410cfcf680b8421e3c22abf31b9297d8e255c3b13acc2890fddc6db6b0',  # noqa: E501
            cache_subdir=path.join('datasets', 'rrc2015'),
            hash_algorithm='sha256',
            extract=True,
            archive_format='zip',
            extract_check_fn=lambda directory: len(glob(path.join(directory, '*.txt'))) == 500  # noqa: E501
        )
        scenes += scenes_from_directories(
            annotations_dir,
            images_dir
        )
    return SceneCollection(
        annotation_config=annotation_config,
        scenes=scenes
    )
