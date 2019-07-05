from typing import Tuple, List
from os import path
import logging

from keras_retinanet import losses as rn_losses
from keras_retinanet.models import backbone as rn_backbone
from keras_retinanet.utils import anchors as rn_anchors
from keras import layers, optimizers
import numpy as np
import cv2

from .detector import Detector
from .. import core
from .. import utils

log = logging.getLogger(__name__)

pretrained = {
    'resnet50': {
        'hash':
        '6518ad56a0cca4d1bd8cbba268dd4e299c7633efe7d15902d5acbb0ba180027c',  # noqa: E501
        'url':
        'https://storage.googleapis.com/miradata/weights/retinanet/resnet50_coco_best_v2.1.0.h5'  # noqa: E501
    }
}


class RetinaNet(Detector):
    """A detector wrapping RetinaNet. All the heavy lifting is done by the
    `keras-retinanet <https://github.com/fizyr/keras-retinanet>`_ package.

    Args:
        annotation_config: The annotation configuration to use for detection
        input_shape: Tuple of (height, width, n_channels)
        pretrained_backbone: Whether to use a pretrained backbone for the model
        pretrained_top: Whether to use the pretrained full model (only
            supported for backbone `resnet50`)
        backbone_name: The name of the backbone to use. See `keras-retinanet`
            for valid options.

    Attributes:
        model: The base Keras model containing the weights for feature
            extraction and bounding box regression / classification model.
            You should use this model for loading and saving weights.
    """

    def __init__(
            self,
            annotation_config: core.AnnotationConfiguration = core.
            AnnotationConfiguration.COCO,  # noqa: E501
            input_shape: Tuple[int, int, int] = (None, None, 3),
            pretrained_backbone: bool = True,
            pretrained_top: bool = False,
            backbone_name='resnet50'):
        if pretrained_top and pretrained_backbone:
            log.info(
                'Disabling imagenet weights, using pretrained full network '
                'instead.')  # noqa: E501
            pretrained_backbone = False
        if pretrained_top:
            assert backbone_name in pretrained, (
                'The selected backbone has no fully trained network. '
                'Acceptable options are: {0}. '  # noqa: E501
                'To resolve this error, choose set pretrained_top=False.'
            ).format(','.join(list(pretrained.keys())))
        self.anchor_params = rn_anchors.AnchorParameters.default
        self.annotation_config = annotation_config
        self.rn_backbone = rn_backbone(backbone_name)
        self.model = self.rn_backbone.retinanet(
            inputs=layers.Input(input_shape),
            num_classes=len(annotation_config),
            num_anchors=self.anchor_params.num_anchors())
        if pretrained_backbone:
            weights_path = self.rn_backbone.download_imagenet()
            log.info('Loading weights from ' + weights_path)
            self.model.load_weights(weights_path, by_name=True)
        if pretrained_top:
            assert annotation_config == core.AnnotationConfiguration.COCO, \
                'To use pretrained_top, annotation config must be core.AnnotationConfiguration.COCO'  # noqa: E501
            pretrained_params = pretrained[backbone_name]
            weights_path = utils.get_file(origin=pretrained_params['url'],
                                          file_hash=pretrained_params['hash'],
                                          cache_subdir=path.join(
                                              'weights', 'retinanet'),
                                          hash_algorithm='sha256',
                                          extract=False)
            self.model.load_weights(weights_path, by_name=True)
        self.compile()

    def invert_targets(self, y, images, threshold=0.5, nms_threshold=0.1):

        boxes = rn_anchors.anchors_for_shape(image_shape=images[0].shape,
                                             anchor_params=self.anchor_params)

        # We filter the third dimension in order to remove the anchor states
        # if `invert_targets` is directly called with the output
        # of `compute_targets`.
        y0 = y[0][:, :, :4]
        y1 = y[1][:, :, :len(self.annotation_config)]

        # The code below reproduces the logic from bbox_transform_inv
        # https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/backend/common.py
        # The only changes are to:
        # * Remove the use of keras because we are not computing the
        #   boxes as part of the network.
        # * Remove the batch dimension of the boxes because they are
        #   from the `anchors_for_shape` function and not part of the
        #   network.

        mean = [0, 0, 0, 0]
        std = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        xl = boxes[:, 0] + (y0[:, :, 0] * std[0] + mean[0]) * width
        yt = boxes[:, 1] + (y0[:, :, 1] * std[1] + mean[1]) * height
        xr = boxes[:, 2] + (y0[:, :, 2] * std[2] + mean[2]) * width
        yb = boxes[:, 3] + (y0[:, :, 3] * std[3] + mean[3]) * height
        w = xr - xl
        h = yb - yt
        xywh = np.concatenate([np.expand_dims(x, -1) for x in [xl, yt, w, h]],
                              axis=2)

        scenes = []
        for boxes, labels, image in zip(xywh, y1, images):
            positive = labels.max(axis=1) > threshold
            if positive.sum() == 0:
                annotations = []
            else:
                boxes = boxes[positive]
                labels = labels[positive]
                classes = np.where(labels.max(axis=0) > threshold)[0]
                predictions = []
                for c in set(classes):
                    class_positive = labels[:, c] > threshold
                    subboxes = boxes[class_positive]
                    sublabels = labels[class_positive, c]
                    bestIdxs = cv2.dnn.NMSBoxes(bboxes=subboxes.tolist(),
                                                scores=sublabels.tolist(),
                                                score_threshold=threshold,
                                                nms_threshold=0.1)[:, 0]
                    subboxes = subboxes[bestIdxs]
                    predictions.extend([b.tolist() + [c] for b in subboxes])
                    annotations = [
                        core.Annotation(selection=core.Selection(
                            [[x, y], [x + w, y + h]]),
                                        category=self.annotation_config[c])
                        for x, y, w, h, c in predictions
                    ]
            scenes.append(
                core.Scene(annotations=annotations,
                           annotation_config=self.annotation_config,
                           image=image))
        return scenes

    def compute_targets(self, collection: core.SceneCollection):
        # Targets are two outputs with shapes
        # (B, N, 5) (four anchor regression targets + anchor state)
        # (B, N, M + 1) (M classification targets + anchor state)
        assert collection.annotation_config == self.annotation_config, \
            'Found incompatible annotation configuration.'
        annotations_group = [{
            'bboxes': b[:, :4],
            'labels': b[:, 4]
        } for b in [s.bboxes() for s in collection]]
        images = collection.images
        anchors = rn_anchors.anchors_for_shape(
            image_shape=images[0].shape, anchor_params=self.anchor_params)
        return list(
            rn_anchors.anchor_targets_bbox(anchors=anchors,
                                           image_group=images,
                                           annotations_group=annotations_group,
                                           num_classes=len(
                                               self.annotation_config),
                                           negative_overlap=0.4,
                                           positive_overlap=0.5))

    def compute_inputs(self, images: List[core.Image]):
        return np.float32(
            [self.rn_backbone.preprocess_image(image) for image in images])

    def compile(self):
        self.model.compile(loss={
            'regression': rn_losses.smooth_l1(),
            'classification': rn_losses.focal()
        },
                           optimizer=optimizers.adam(lr=1e-5, clipnorm=0.001))

    def freeze_backbone(self):
        output_names = [
            'regression_submodel', 'classification_submodel', 'P3', 'P4', 'P5',
            'P6', 'P7'
        ]
        for l in self.model.layers:
            if (l.name in output_names or l.name in self.model.output_names):
                log.info('Not freezing ' + l.name)
            else:
                log.info('Freezing ' + l.name)
                l.trainable = False
        self.compile()

    def unfreeze_backbone(self):
        """Unfreeze the body of the model, making all layers trainable."""
        for l in self.model.layers:
            l.trainable = True
        self.compile()
