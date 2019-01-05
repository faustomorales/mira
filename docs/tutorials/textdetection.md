# Training a text detection model

This is a work-in-progress.

```
from mira import core, detectors, datasets
from imgaug import augmenters as iaa

training = datasets.load_icdar2015(subset='train')
validation = datasets.load_icdar2015(subset='test')

augmenter = iaa.Sequential(
    iaa.CropToFixedSize(width=512, height=512)
)

detector = detectors.EASTTextDetector(
    pretrained_top=True,
    annotation_config=training.annotation_config
)

detector.freeze_backbone()

detector.train(
    training=training,
    validation=validation,
    batch_size=1,
    steps_per_epoch=50,
    epochs=100,
    augmenter=augmenter,
    train_shape=(128, 128, 3)
)
```