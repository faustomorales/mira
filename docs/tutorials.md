# Tutorials

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

.. image:: _static/example_browsing.png
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

.. image:: _static/example_augmentation.png
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

detector_ed = detectors.EfficientDet(pretrained_top=True)

# Pick an example scene
scene = dataset[5]

# Set up side-by-side plots
fig, (ax_ed, ax_yolo) = plt.subplots(ncols=2, figsize=(10, 5))
ax_ed.set_title('EfficientDet')
ax_yolo.set_title('YOLOv3')

# We get predicted scenes from each detector. Detectors return
# lists of annotations for a given image. So we can just replace
# (assign) those new annotations to the scene to get a new scene
# reflecting the detector's prediction.
predicted_ed = scene.assign(
    annotations=detector_ed.detect(scene.image),
    annotation_config=detector_ed.annotation_config
)
predicted_yolo = scene.assign(
    annotations=detector_yolo.detect(scene.image, threshold=0.4),
    annotation_config=detector_yolo.annotation_config
)

# Plot both predictions. The calls to annotation() get us
# an image with the bounding boxes drawn.
_ = predicted_ed.annotated().show(ax=ax_ed)
_ = predicted_yolo.annotated().show(ax=ax_yolo)
```

.. image:: _static/example_simple_od.png
    :alt: annotated image

## Transfer Learning
This example inspired by [the TensorFlow object detection tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md).

```
from mira import datasets, detectors
from imgaug import augmenters as iaa
from keras import callbacks

# Load the Oxford pets datasets with a class
# for each breed.
dataset = datasets.load_oxfordiiitpets(breed=True)

# Load YOLO with pretrained backbone. We'll
# use the annotation configuration for our
# new task.
detector = detectors.YOLOv3(
    input_shape=(256, 256, 3),
    pretrained_top=False,
    pretrained_backbone=True,
    annotation_config=dataset.annotation_config,
    size='tiny'
)

# Split our dataset into training, validation,
# and test.
trainval, testing = dataset.train_test_split(
    train_size=0.7, test_size=0.3
)
training, validation = trainval.train_test_split(
    train_size=0.66, test_size=0.33
)

# Create an augmenter
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

# Make a callback to stop the training job
# early if we plateau on the validation set.
cbs = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=50,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )
]

# Run training job
detector.train(
    training=training,
    validation=validation,
    steps_per_epoch=50,
    epochs=1000,
    batch_size=8,
    augmenter=augmenter,
    callbacks=cbs
)
```

## Training on COCO 2014

Start by downloading all of the COCO images. If you haven't already, install `gsutil`, which will make downloading the images a snap.

```shell
curl https://sdk.cloud.google.com | bash
mkdir coco
cd coco
mkdir images
gsutil -m rsync gs://images.cocodataset.org/train2014 images
gsutil -m rsync gs://images.cocodataset.org/val2014 images
gsutil -m rsync gs://images.cocodataset.org/test2014 images
```

Now download the annotations. The following will create an `annotations` folder inside of your `coco` folder.
```shell
curl -LO  http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

Let's load the dataset.

```python
from mira import datasets

training = datasets.load_coco(
    annotations_file='coco/annotations/instances_train2014.json',
    image_dir='coco/images'
)
validation = datasets.load_coco(
    annotations_file='coco/annotations/instances_val2014.json',
    image_dir='coco/images'
)
```

And now we can train a detector.

```python
from mira import detectors

detector = detectors.YOLOv3(
    annotation_config=training.annotation_config
)
detector.train(
    training=training,
    validation=validation,
    epochs=1000,
    steps_per_epoch=50
    train_shape=(512, 512)
)
```
