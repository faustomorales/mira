# Quickstart

## Logging
To see logs, use the `logging` API. The logs can be
useful for keeping track of what is happening behind
the scenes.

```python
import logging
logging.basicConfig()
logging.getLogger().setLevel('INFO')
```

## Browsing image data

```python
from mira import datasets
coco = datasets.load_coco2017(subset='val')
coco[26].annotated().show()
```

.. image:: ../_static/example_browsing.png
    :alt: basic image


## Augmentation
Augmentation can be kind of a pain for
object detection sometimes. But `imgaug` makes
it pretty easy to build augmentation pipelines
and mira uses them to transform images as well
as annotations.

```python
from mira import datasets
from imgaug import augmenters as iaa

dataset = datasets.load_voc2012(subset='val')
scene = dataset[15]
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    iaa.Affine(
        scale=(0.9, 1.1), # scale between 90% and 110% of original
        translate_percent=(0.1, -0.1), # Translate +/- 10% of image size
        rotate=(-5, 5),  # rotate -5 degrees to 5 degrees
        cval=255
    )
])

fig, (ax_original, ax_augmenter) = plt.subplots(ncols=2, figsize=(10, 5))
ax_original.set_title('Original')
ax_augmenter.set_title('Augmented')

scene.annotated().show(ax=ax_original)
scene.augment(augmenter).annotated().show(ax=ax_augmenter)
```

.. image:: ../_static/example_augmentation.png
    :alt: augmented image


## Basic object detection

The example below shows how easy it is to
do object detection using the common API
for detectors.

```python
from mira import datasets, detectors
import matplotlib.pyplot as plt

# Load the VOC dataset (use the validation
# split to save time)
dataset = datasets.load_voc2012(subset='val')

# Load YOLO with pretrained layers. It is
# set up to use COCO labels.
detector_yolo = detectors.YOLOv3(
    input_shape=(256, 256, 3),
    pretrained_top=True,
    size='tiny'
)

# Load RetinaNet with pretrained layers.
# It also uses COCO labels.
detector_rn = detectors.RetinaNet(
    input_shape=(256, 256, 3),
    pretrained_top=True,
    backbone_name='resnet50'
)

# Pick an example scene
scene = dataset[5]

# Set up side-by-side plots
fig, (ax_rn, ax_yolo) = plt.subplots(ncols=2, figsize=(10, 5))
ax_rn.set_title('RetinaNet')
ax_yolo.set_title('YOLOv3')

# We get predicted scenes from each detector. Detectors return
# lists of annotations for a given image. So we can just replace
# (assign) those new annotations to the scene to get a new scene
# reflecting the detector's prediction.
predicted_rn = scene.assign(
    annotations=detector_rn.detect(scene.image),
    annotation_config=detector_rn.annotation_config
)
predicted_yolo = scene.assign(
    annotations=detector_yolo.detect(scene.image, threshold=0.4),
    annotation_config=detector_yolo.annotation_config
)

# Plot both predictions. The calls to annotation() get us
# an image with the bounding boxes drawn.
_ = predicted_rn.annotated().show(ax=ax_rn)
_ = predicted_yolo.annotated().show(ax=ax_yolo)
```

.. image:: ../_static/example_simple_od.png
    :alt: annotated image