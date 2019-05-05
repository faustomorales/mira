# Transfer Learning
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