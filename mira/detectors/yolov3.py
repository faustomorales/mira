from functools import wraps
from typing import Tuple, List
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, layers, models, optimizers

from .detector import Detector
from ..utils import compute_overlap
from .. import core
from ..utils import get_file, get_datadir_base
from .utils import convert_darknet_weights

"""YOLOv3 Detector

The model architecture here is taken directly from
[this previous work](https://github.com/qqwweee/keras-yolo3). I would
have simply used the original if it were a pip-installable package. The
prediction logic and loss functions are modified to match my stylistic
preferences for documentation, and I take responsibility for any
implementation errors.

The following are a few variable definitions for reference:

- tx, ty, tw, th     : The bounding box outputs from the base model
- txs, tys, twe, the : The bounding box outputs from the training model
- bx, by, bw, bh     : The bounding box outputs from the prediction model
- sx, sy             : The stride of a given box in an output (dependent on the
                       model architecture).
- cx, cy             : The prior offset of a prediction box
- pw, ph             : The prior width and height of a prediction box

In all cases, x and y refer to the center of a given bounding box.

The following formulas govern the relationship between the above variables.
- txs = (bx - cx) / sx
- tys = (by - cy) / sy
- twe = bw / pw
- the = bh / ph
- txs = sigmoid(tx)
- tys = sigmoid(ty)
- twe = exp(tw)
- the = exp(th)

To clarify, txs and tys are factors of the stride to adjust the bounding box.
So txs == 1 corresponds with moving the box center one full stride to the right
from its original position.

Similarly, twe is a factor of the prior width. So twe == 2 corresponds with
doubling the width of the original bounding box.

The main difference between YOLO and Tiny YOLO is that that Tiny YOLO only
uses two scales (two anchor groups with three anchors in each) while YOLO uses
three scales (three anchor groups with three anchors in each)
"""

log = logging.getLogger(__name__)


YOLO_ANCHOR_GROUPS = {
    'tiny': [
        np.array([[81, 82], [135, 169], [344, 319]]),
        np.array([[10, 14], [23, 27], [37, 58]])
    ],
    'full': [
        np.array([[116, 90], [156, 198], [373, 326]]),
        np.array([[30, 61], [62, 5], [59, 119]]),
        np.array([[10, 13], [16, 30], [33, 23]])
    ]
}

YOLO_FILE_CONFIG = {
    'tiny': {
        'weights_fname': 'yolov3-tiny.weights',
        'weights_url': 'https://pjreddie.com/media/files/yolov3-tiny.weights',
        'weights_hash': 'dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c',   # noqa: E501
        'cfg_fname': 'yolov3-tiny.cfg',
        'cfg_url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',  # noqa: E501
        'cfg_hash': '84eb7a675ef87c906019ff5a6e0effe275d175adb75100dcb47f0727917dc2c7',  # noqa: E501
        'converted_fname': 'yolov3_tiny.h5',
        'converted_notop_fname': 'yolov3_tiny_notop.h5',
    },
    'full': {
        'weights_fname': 'yolov3.weights',
        'weights_url': 'https://pjreddie.com/media/files/yolov3.weights',
        'weights_hash': '523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297',  # noqa: E501
        'cfg_fname': 'yolov3.cfg',
        'cfg_url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',  # noqa: E501
        'cfg_hash': '22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953',  # noqa: E501
        'converted_fname': 'yolov3.h5',
        'converted_notop_fname': 'yolov3_notop.h5',
    }
}


@wraps(layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D.
    All arguments passed to `layers.Conv2D`"""

    defaults = {
        'kernel_regularizer': regularizers.l2(5e-4),
        'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same',
    }
    kwargs = {**defaults, **kwargs}
    return layers.Conv2D(*args, **kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    All arguments passed to DarknetConv2D"""
    defaults = {
        'use_bias': False
    }
    kwargs = {**defaults, **kwargs}

    def make_layer(x):
        x = DarknetConv2D(*args, **kwargs)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    return make_layer


def resblock_body(x, num_filters, num_blocks, name):
    """A series of resblocks starting with a downsampling
    Convolution2D.

    Args:
        x: Input layer
        num_filters: number of filters for DarknetConv2D
        num_blocks: Number of blocks to add
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(
        filters=num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        name=name+'_c2d_bn_leaky'
    )(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(
            filters=num_filters // 2,
            kernel_size=(1, 1),
            name=name+'_block{0}_1x1_c2d_bn_leaky'.format(i)
        )(x)
        y = DarknetConv2D_BN_Leaky(
            filters=num_filters,
            kernel_size=(3, 3),
            name=name+'_block{0}_3x3_c2d_bn_leaky'.format(i)
        )(y)
        x = layers.Add(name=name+'_block{0}_add'.format(i))([x, y])
    return x


def make_input_layer(x, num_filters, name):
    xi = DarknetConv2D_BN_Leaky(filters=num_filters, kernel_size=(1, 1), name=name+'_c2d_bn_leaky1')(x)  # noqa: E501
    xi = DarknetConv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), name=name+'_c2d_bn_leaky2')(xi)  # noqa: E501
    xi = DarknetConv2D_BN_Leaky(filters=num_filters, kernel_size=(1, 1), name=name+'_c2d_bn_leaky3')(xi)  # noqa: E501
    xi = DarknetConv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), name=name+'_c2d_bn_leaky4')(xi)  # noqa: E501
    xi = DarknetConv2D_BN_Leaky(filters=num_filters, kernel_size=(1, 1), name=name+'_c2d_bn_leaky5')(xi)  # noqa: E501
    x = DarknetConv2D_BN_Leaky(filters=num_filters*2, kernel_size=(3, 3), name=name+'_c2d_bn_leaky_out')(xi)  # noqa: E501
    return x, xi


def get_backbone(inputs, size):
    """Create YOLO_V3 model CNN body in Keras"""
    assert size in ['tiny', 'full'], 'Size must be `tiny` or `full`'
    if size == 'tiny':
        x = DarknetConv2D_BN_Leaky(filters=16, kernel_size=(3, 3))(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # noqa: E501
        x = DarknetConv2D_BN_Leaky(filters=32, kernel_size=(3, 3))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # noqa: E501
        x = DarknetConv2D_BN_Leaky(filters=64, kernel_size=(3, 3))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # noqa: E501
        x = DarknetConv2D_BN_Leaky(filters=128, kernel_size=(3, 3))(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # noqa: E501
        x = DarknetConv2D_BN_Leaky(filters=256, kernel_size=(3, 3))(x)

        x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # noqa: E501
        x1 = DarknetConv2D_BN_Leaky(filters=512, kernel_size=(3, 3))(x1)
        x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x1)  # noqa: E501
        x1 = DarknetConv2D_BN_Leaky(filters=1024, kernel_size=(3, 3))(x1)
        x1 = DarknetConv2D_BN_Leaky(filters=256, kernel_size=(1, 1))(x1)

        x2 = DarknetConv2D_BN_Leaky(filters=128, kernel_size=(1, 1))(x1)
        x2 = layers.UpSampling2D(2)(x2)
        x2 = layers.Concatenate()([x2, x])
        x2 = DarknetConv2D_BN_Leaky(filters=256, kernel_size=(3, 3))(x2)

        x1 = DarknetConv2D_BN_Leaky(filters=512, kernel_size=(3, 3))(x1)
        outputs = [x1, x2]
    else:
        conv1 = DarknetConv2D_BN_Leaky(32, (3, 3), name='conv_bn_leaky1')(inputs)  # noqa: E501
        rb1 = resblock_body(conv1, 64, 1, name='resblock1')
        rb2 = resblock_body(rb1, 128, 2, name='resblock2')
        rb3 = resblock_body(rb2, 256, 8, name='resblock3')
        rb4 = resblock_body(rb3, 512, 8, name='resblock4')
        rb5 = resblock_body(rb4, 1024, 4, name='resblock5')

        x1, x1i = make_input_layer(rb5, 512, name='x1')

        x2 = DarknetConv2D_BN_Leaky(filters=256, kernel_size=(1, 1))(x1i)
        x2 = layers.UpSampling2D(2, name='upsampling_x2')(x2)

        x2 = layers.Concatenate()([x2, rb4])
        x2, x2i = make_input_layer(x2, 256, name='x2')

        x3 = DarknetConv2D_BN_Leaky(filters=128, kernel_size=(1, 1))(x2i)
        x3 = layers.UpSampling2D(2, name='upsampling_x3')(x3)

        x3 = layers.Concatenate()([x3, rb3])
        x3, x3i = make_input_layer(x3, 128, name='x3')
        outputs = [x1, x2, x3]

    return models.Model(inputs, outputs)


def get_top(backbone, num_anchors_per_output, num_classes):
    filters_per_anchor = (num_classes+5)
    out_filters = num_anchors_per_output*filters_per_anchor
    outputs = []
    for i, x in enumerate(backbone.outputs):
        y = DarknetConv2D(
            filters=out_filters,
            kernel_size=(1, 1),
            name='prediction_group_{0}'.format(i+1)
        )(x)
        concatenate = []
        for n in range(num_anchors_per_output):
            # Below are txs, tys, twe, the, tys, and class probabilites
            startIdx = n*filters_per_anchor
            txsi = startIdx + 0
            tysi = startIdx + 1
            twei = startIdx + 2
            thei = startIdx + 3
            si = startIdx + 4
            csi = startIdx + 5
            csj = csi + num_classes

            def apply_activation(y):
                return layers.Concatenate()([
                    layers.Activation('sigmoid')(y[..., txsi:txsi+1]),
                    layers.Activation('sigmoid')(y[..., tysi:tysi+1]),
                    layers.Activation('exponential')(y[..., twei:twei+1]),
                    layers.Activation('exponential')(y[..., thei:thei+1]),
                    layers.Activation('sigmoid')(y[..., si:si+1]),
                    # As per section 2.2 of the paper, we use independent
                    # logits for each class, instead of softmax.
                    layers.Activation('sigmoid')(y[..., csi:csj])
                ])
            group = layers.Lambda(apply_activation)(y)
            concatenate.append(group)
        y = layers.Concatenate(
            name='anchor_group_{0}'.format(i+1)
        )(concatenate)
        outputs.append(y)
    return models.Model(backbone.inputs, outputs)


def anchors_for_shape(input_shape, anchor_groups):
    """Compute the anchors for a given output shape and anchor
    groups.

    Returns:
        List of anchor coordinates. Each entry in the list is
        of shape (steps_y, steps_x, len(seeds), 6). The entries
        are the base x, y, width, height, horizontal stride, and
        vertical stride, respectively.
    """
    image_h, image_w = input_shape
    anchors = []
    for seeds, stride in zip(anchor_groups, [32, 16, 8]):
        steps_y, steps_x = [int(s / stride) for s in input_shape]
        cx = np.reshape(
            np.arange(start=0, stop=steps_x, dtype='float32'),
            (1, steps_x, 1, 1)
        ) * (image_w / steps_x)
        cy = np.reshape(
            np.arange(start=0, stop=steps_y, dtype='float32'),
            (steps_y, 1, 1, 1)
        ) * (image_h / steps_y)
        xywhs = np.zeros((steps_y, steps_x, len(seeds), 6))
        pw = seeds[:, 0:1]
        ph = seeds[:, 1:2]
        sx = image_w / steps_x
        sy = image_h / steps_y
        xywhs[..., 0:1] = cx
        xywhs[..., 1:2] = cy
        xywhs[..., 2:3] = pw
        xywhs[..., 3:4] = ph
        xywhs[..., 4:5] = sx
        xywhs[..., 5:6] = sy
        anchors.append(xywhs)
    return anchors


def make_loss(num_classes):
    num_features = 4+1+num_classes

    def loss(y_true, y_pred):
        """Loss function for YOLO. y_true and y_pred
        have shape (batchSize, y_steps, x_steps, n_features). This
        loss function is used for each output separately.
        """
        num_anchor_seeds = int(y_pred.shape[-1].value / num_features)
        loss_objectness = 0
        loss_coords = 0
        loss_labels = 0

        for j in range(num_anchor_seeds):
            startIdx = j*num_features
            coordsi = startIdx
            coordsj = coordsi + 4
            labelsi = startIdx + 5
            labelsj = labelsi + num_classes
            objectnessi = startIdx + 4

            # For object state, -1 for ignore, 0 for background, 1
            # for object
            coords_true = y_true[:, :, :, coordsi:coordsj]
            labels_true = y_true[:, :, :, labelsi:labelsj]
            object_state_true = y_true[:, :, :, objectnessi]

            coords_pred = y_pred[:, :, :, coordsi:coordsj]
            labels_pred = y_pred[:, :, :, labelsi:labelsj]
            object_state_pred = y_pred[:, :, :, objectnessi]

            # Calculate objectness loss, which applies to
            # all non-ignore anchors
            indices = tf.where(K.not_equal(object_state_true, -1))
            true = tf.gather_nd(object_state_true, indices)
            pred = tf.gather_nd(object_state_pred, indices)
            loss_objectness += K.sum(K.square(true - pred))

            # Calculate coordinate loss which applies only
            # to positive anchors
            indices = tf.where(K.equal(object_state_true, 1))
            true = tf.gather_nd(coords_true, indices)
            pred = tf.gather_nd(coords_pred, indices)
            loss_coords += K.sum(K.square(true - pred))

            # Calculate classification loss, which applies
            # only to positive anchors. Indices same as above.
            true = tf.gather_nd(labels_true, indices)
            pred = tf.gather_nd(labels_pred, indices)
            loss_labels += K.sum(K.square(true - pred))

        # Calculate total loss
        return loss_objectness + loss_coords + loss_labels

    return loss


def convert_anchors(anchors_xywhs):
    """Convert anchors from xc, yc, w, h to
    x1, y1, x2, y2"""
    anchors_xyxy = np.zeros(
        (len(anchors_xywhs), 4)
    )
    anchors_xyxy[:, 0:2] = anchors_xywhs[:, 0:2] - (anchors_xywhs[:, 2:4] / 2)
    anchors_xyxy[:, 2:4] = anchors_xywhs[:, 0:2] + (anchors_xywhs[:, 2:4] / 2)
    return anchors_xyxy


class YOLOv3(Detector):
    """A detector wrapping
    `YOLOv3 <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_.
    Most of the implementation is borrowed from
    `keras-yolov3 <https://github.com/qqwweee/keras-yolo3>`_.

    Args:
        annotation_config: The annotation configuration to use for detection
        input_shape: Tuple of (height, width, n_channels)
        size: One of `tiny` or `full`.
        pretrained_backbone: Whether to use a pretrained backbone for the model
        pretrained_top: Whether to use the pretrained full model (only
            supported for where `annotation_config` is
            `AnnotationConfiguration.COCO`)

    Attributes:
        model: The base Keras model containing the weights for feature
            extraction and bounding box regression / classification model.
            You should use this model for loading and saving weights.
        backbone: The backbone for the model (all layers except for final
            classification and regression).
    """
    def __init__(
        self,
        annotation_config: core.AnnotationConfiguration=core.AnnotationConfiguration.COCO,  # noqa: E501
        input_shape: Tuple[int, int]=(None, None, 3),
        size: str='tiny',
        pretrained_backbone: bool=True,
        pretrained_top: bool=False
    ):
        assert (
            not pretrained_top or
            annotation_config == core.AnnotationConfiguration.COCO
        ), (
            'pretrained_top requires annotation configuration '
            'to be AnnotationConfiguration.COCO'
        )
        self.anchor_groups = YOLO_ANCHOR_GROUPS[size]
        self.annotation_config = annotation_config
        self.size = size
        self.backbone = get_backbone(
            inputs=layers.Input(input_shape),
            size=size
        )
        self.model = get_top(
            backbone=self.backbone,
            num_classes=len(annotation_config),
            num_anchors_per_output=len(self.anchor_groups[0])
        )
        if pretrained_backbone and pretrained_top:
            log.warning(
                'pretrained_top makes pretrained_backbone '
                'redundant. Disabling pretrained_backbone.'
            )
            pretrained_backbone = False
        if pretrained_backbone or pretrained_top:
            weights_path = os.path.join(
                get_datadir_base(),
                'weights',
                'yolov3',
                YOLO_FILE_CONFIG[size][
                    'converted_notop_fname' if pretrained_backbone
                    else 'converted_fname'
                ]
            )
            if not os.path.isfile(weights_path):
                log.warning(
                    "Could not find weights file. Attempting to build. "
                    "If error occurs, please execute "
                    "YOLOv3.make_pretrained_weight_files(sizes=['{0}'])".format(  # noqa: E501
                        size
                    ) + " separately first."
                )
                YOLOv3.make_pretrained_weight_files(sizes=[size])
            if pretrained_backbone:
                self.backbone.load_weights(weights_path)
            if pretrained_top:
                self.model.load_weights(weights_path)
        self.compile()

    def invert_targets(
        self,
        y: List[np.ndarray],
        images: List[core.Image],
        threshold: float=0.5
    ) -> core.SceneCollection:
        num_features = 4+1+len(self.annotation_config)
        output_anchors = anchors_for_shape(
            input_shape=images[0].shape[:2],
            anchor_groups=self.anchor_groups
        )
        batch_size = len(y[0])
        processed = []
        for output, anchors in zip(y, output_anchors):
            output = output.reshape((batch_size, -1, num_features))
            anchors = anchors.reshape((1, -1, 6))
            cx, cy, pw, ph, sx, sy = [anchors[:, :, i:i+1] for i in range(6)]
            txs, tys, twe, the, c = [output[:, :, i:i+1] for i in range(5)]
            cc = output[:, :, 5:]
            bx = txs*sx + cx
            by = tys*sy + cy
            bw = pw*twe
            bh = ph*the
            processed.append(np.concatenate([bx, by, bw, bh, c, cc], axis=-1))
        box_groups = [
            np.concatenate([p[idx] for p in processed])
            for idx in range(batch_size)
        ]

        scenes = []
        for boxes, image in zip(box_groups, images):
            boxes = boxes[boxes[:, 4] > threshold]
            annotations = []
            for xi, yi, w, h, s, c in zip(
                boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                boxes[:, 4], boxes[:, 5:].argmax(axis=1)
            ):
                annotations.append(
                    core.Annotation(
                        selection=core.Selection(
                            points=[
                                (xi-w/2, yi-h/2),
                                (xi+w/2, yi+h/2),
                            ]
                        ),
                        category=self.annotation_config[c],
                        score=s
                    )
                )
            scenes.append(core.Scene(
                annotation_config=self.annotation_config,
                annotations=annotations,
                image=image
            ))
        return scenes

    def compile(self):
        # Each anchor group has four times as many anchors as the prior one
        n = len(self.anchor_groups)
        total = sum(4**i for i in range(n))
        weights = [4**i / total for i in range(n)]
        self.model.compile(
            loss=make_loss(
                num_classes=len(self.annotation_config)
            ),
            loss_weights=weights,
            optimizer=optimizers.Adam()
        )

    def compute_inputs(self, images: List[core.Image]):
        return np.float32([image.scaled(0, 1) for image in images])

    def compute_targets(self, collection: core.SceneCollection):
        """Preprocess true boxes to training input format

        Args:
            collection: The collection of scenes for which to build output

        Returns:
            y_true: list of array, shape like yolo_outputs, xywh are relative
                values.
        """
        ignore_overlap = 0.5
        anchors = anchors_for_shape(
            input_shape=collection.images[0].shape[:2],
            anchor_groups=self.anchor_groups
        )
        results = [[] for group in anchors]
        anchors_xywh = np.concatenate(
            [a.reshape(-1, 6) for a in anchors],
            axis=0
        )
        anchors_xyxy = convert_anchors(anchors_xywh)
        for scene in collection.scenes:
            annotations = scene.bboxes()
            y_true = np.zeros(
                (len(anchors_xyxy), 4+1+len(self.annotation_config))
            )
            if len(annotations) > 0:
                overlaps = compute_overlap(
                    boxes=anchors_xyxy.astype(np.float64),
                    query_boxes=annotations.astype(np.float64)
                )

                # Set objectness targets
                # The objectness score should be:
                #     1, if the bounding box prior overlaps a ground
                #        truth object by more than any other bounding box prior
                #        and overlaps by more than threshold
                #     0, if the bounding box does not overlap any ground truth
                #        object by more than threshold
                #     ignored (-1), if the bounding box overlaps a ground truth
                #        object by more than threshold, but is not the box of
                #        maximum overlap
                ignored_index = (overlaps >= ignore_overlap).any(axis=1)
                positive_indices = overlaps.argmax(axis=0)
                ignored_index[positive_indices] = False
                background_index = ~ignored_index
                background_index[positive_indices] = False
                y_true[positive_indices, 4] = 1
                y_true[ignored_index, :] = 0
                y_true[ignored_index, 4] = -1
                y_true[background_index, :] = 0
                y_true[background_index, 4] = 0

                # Set class targets
                y_true[positive_indices, 5:] = 0
                y_true[positive_indices, 5+annotations[:, 4]] = 1

                # Set x, y, w, h targets, or more formally,
                # the txs, tys, twe, and the targets (see formulas
                # for additional details).
                bw = annotations[:, 2] - annotations[:, 0]
                bh = annotations[:, 3] - annotations[:, 1]
                bx = annotations[:, 0] + (bw / 2)
                by = annotations[:, 1] + (bh / 2)
                cx = anchors_xywh[positive_indices, 0]
                cy = anchors_xywh[positive_indices, 1]
                pw = anchors_xywh[positive_indices, 2]
                ph = anchors_xywh[positive_indices, 3]
                sx = anchors_xywh[positive_indices, 4]
                sy = anchors_xywh[positive_indices, 5]
                y_true[positive_indices, 0] = (bx - cx) / sx
                y_true[positive_indices, 1] = (by - cy) / sy
                y_true[positive_indices, 2] = bw / pw
                y_true[positive_indices, 3] = bh / ph

            # The loss function should:
            # - Penalize 'positive' (1) boxes for coordinate, objectness,
            #       and class predictions
            # - Penalize 'background' (0) boxes for objectness predictions
            # - Penalize 'ignore' (-1) boxes for nothing
            anchorIdx = 0
            for i, group in enumerate(anchors):
                n_anchors = np.product(group.shape[:3])
                selected = y_true[anchorIdx:anchorIdx+n_anchors].reshape(
                    group.shape[0], group.shape[1], -1
                )
                results[i].append(
                    selected
                )
                anchorIdx += n_anchors
        results = [np.array(r) for r in results]
        return results

    @staticmethod
    def make_pretrained_weight_files(sizes=['tiny', 'full']):
        for size in sizes:
            config = YOLO_FILE_CONFIG[size]
            log.info('Getting pretrained weights')
            converted_path = os.path.join(
                get_datadir_base(),
                'weights',
                'yolov3',
                config['converted_fname']
            )
            converted_path_notop = converted_path.replace(
                config['converted_fname'],
                config['converted_notop_fname']
            )
            if not os.path.isfile(converted_path):
                log.info('Converting darknet weights to keras.')
                log.info('Getting darknet model weights file.')
                weights_fpath = get_file(
                    fname=config['weights_fname'],
                    origin=config['weights_url'],
                    file_hash=config['weights_hash'],
                    cache_subdir=os.path.join('weights', 'yolov3'),
                    hash_algorithm='sha256'
                )
                log.info('Getting model configuration')
                cfg_path = get_file(
                    fname=config['cfg_fname'],
                    origin=config['cfg_url'],
                    file_hash=config['cfg_hash'],
                    cache_subdir=os.path.join('weights', 'yolov3'),
                    hash_algorithm='sha256'
                )
                convert_darknet_weights(
                    config_path=cfg_path,
                    weights_path=weights_fpath,
                    output_path=converted_path,
                    weights_only=True
                )
            else:
                log.info('Found existing converted weights.')
            log.info('Creating notop and complete weight files.')
            backbone = get_backbone(
                inputs=layers.Input((None, None, 3)),
                size=size
            )
            base_model = get_top(
                backbone=backbone,
                num_anchors_per_output=len(YOLO_ANCHOR_GROUPS[size][0]),
                num_classes=len(core.AnnotationConfiguration.COCO)
            )
            base_model.load_weights(converted_path)
            backbone.save_weights(converted_path_notop)