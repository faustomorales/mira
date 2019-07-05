from typing import Tuple, List
import logging

from tensorflow.keras import layers, regularizers, models, optimizers
from tensorflow.keras import backend as K
from nms import nms
import numpy as np
import cv2

from .. import core
from .detector import Detector

log = logging.getLogger(__name__)


class ScaleShift(layers.Layer):
    """Implement Scale / Shift opertaion from PVANet paper.

        out = in * gamma + beta,

    'gamma' and 'beta' are the learned weights and biases.

    Args:
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
    """
    def __init__(self, weights=None, **kwargs):
        self.initial_weights = weights
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (int(input_shape[-1]),)

        # Tensorflow >= 1.0.0 compatibility
        self._gamma = K.variable(np.ones(shape), name='{}_gamma'.format(self.name))  # noqa: E501
        self._beta = K.variable(np.zeros(shape), name='{}_beta'.format(self.name))  # noqa: E501
        self._trainable_weights = [self._gamma, self._beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return self._gamma * x + self._beta


def inception(
    x,
    f1x1: int,
    f3x3: Tuple[int, int],
    f5x5: Tuple[int, int, int],
    fOut: int,
    prefix: str,
    fPool: int=None,
    stride: int=1,
    outputBn=False
):
    """Make Inception layer according to the description in Figure 1
    in https://arxiv.org/pdf/1608.08021.pdf.

    """
    assert stride == 1 or fPool is not None, \
        'If stride != 1, fPool must be provided'
    assert (
        fOut == x.get_shape().as_list()[-1] or
        stride != 1
    ), 'The number of input filters must equal output filters when stride is 1.'  # noqa: E501
    x = layers.BatchNormalization(name=prefix+'/incep/bn')(x)
    x = layers.Activation('relu', name=prefix+'/incep/relu')(x)

    x1x1 = layers.Conv2D(f1x1, 1, strides=stride, name=prefix+'/incep/0/conv', use_bias=False)(x)  # noqa: E501
    x1x1 = layers.BatchNormalization(name=prefix+'/incep/0/bn')(x1x1)
    x1x1 = layers.Activation('relu', name=prefix+'/incep/0/relu')(x1x1)

    x3x3 = layers.Conv2D(f3x3[0], 1, strides=stride, name=prefix+'/incep/1_reduce/conv', use_bias=False)(x)  # noqa: E501
    x3x3 = layers.BatchNormalization(name=prefix+'/incep/1_reduce/bn')(x3x3)
    x3x3 = layers.Activation('relu', name=prefix+'/incep/1_reduce/relu')(x3x3)
    x3x3 = layers.Conv2D(f3x3[0], 3, name=prefix+'/incep/1_0/conv', padding='same', use_bias=False)(x3x3)  # noqa: E501
    x3x3 = layers.BatchNormalization(name=prefix+'/incep/1_0/bn')(x3x3)
    x3x3 = layers.Activation('relu', name=prefix+'/incep/1_0/relu')(x3x3)

    x5x5 = layers.Conv2D(f5x5[0], 1, strides=stride, name=prefix+'/incep/2_reduce/conv', use_bias=False)(x)  # noqa: E501
    x5x5 = layers.BatchNormalization(name=prefix+'/incep/2_reduce/bn')(x5x5)
    x5x5 = layers.Activation('relu', name=prefix+'/incep/2_reduce/relu')(x5x5)
    x5x5 = layers.Conv2D(f5x5[1], 3, name=prefix+'/incep/2_0/conv', padding='same', use_bias=False)(x5x5)  # noqa: E501
    x5x5 = layers.BatchNormalization(name=prefix+'/incep/2_0/bn')(x5x5)
    x5x5 = layers.Activation('relu', name=prefix+'/incep/2_0/relu')(x5x5)
    x5x5 = layers.Conv2D(f5x5[2], 3, name=prefix+'/incep/2_1/conv', padding='same', use_bias=False)(x5x5)  # noqa: E501
    x5x5 = layers.BatchNormalization(name=prefix+'/incep/2_1/bn')(x5x5)
    x5x5 = layers.Activation('relu', name=prefix+'/incep/2_1/relu')(x5x5)

    if stride == 2:
        xPool = layers.MaxPooling2D(3, strides=2, name=prefix+'/incep/pool', padding='same')(x)  # noqa: E501
        xPool = layers.Conv2D(fPool, 1, name=prefix+'/incep/poolproj/conv', use_bias=False)(xPool)  # noqa: E501
        xPool = layers.BatchNormalization(name=prefix+'/incep/poolproj/bn')(xPool)  # noqa: E501
        xPool = layers.Activation('relu', name=prefix+'/incep/poolproj/relu')(xPool)  # noqa: E501
        outs = [x1x1, x3x3, x5x5, xPool]
    elif stride == 1:
        outs = [x1x1, x3x3, x5x5]
    else:
        raise NotImplementedError

    y = layers.Concatenate(name=prefix+'/incep/concat')(outs)
    y = layers.Conv2D(fOut, 1, name=prefix+'/out/conv')(y)

    xRes = x
    if stride != 1:
        xRes = layers.Conv2D(
            fOut, 1, strides=stride, name=prefix+'/proj'
        )(xRes)
    y = layers.Add(name=prefix+'/add')([y, xRes])
    if outputBn:
        y = layers.BatchNormalization(name=prefix+'/last_bn')(y)
        y = layers.Activation('relu', name=prefix+'/last_relu')(y)
    return y


def crelu(
    x, k: int, fcReLU: int, prefix: str, stride: int=1,
    fIn: int=None, fOut: int=None, residual: bool=True,
    input_bn=True
):
    """Make c.ReLU according to the description in Figure 1 in
    https://arxiv.org/pdf/1608.08021.pdf.

    Args:
        x: The input layer
        k: The kernel size for the c.ReLU block
        fcReLU: The number of filters for the c.ReLU block
        prefix: The prefix for the layers in the block
        stride: The stride for the c.ReLU block
        fIn: The number of filters on the initial 1x1 convolution
        fOut: The number of filters for the output 1x1 convolution
        residual: If True, the input layer (x) is added as a residual.
        input_bn: Whether to apply batch normalization and activation
            to input.

    Returns:
        The output layer
    """
    xRes = x
    if input_bn:
        x = layers.BatchNormalization(name=prefix+'/1/bn')(x)
        x = layers.Activation('relu', name=prefix+'/1/relu')(x)
    if fIn is not None:
        x = layers.Conv2D(fIn, 1, name=prefix+'/1/conv')(x)
        x = layers.BatchNormalization(name=prefix+'/2/bn')(x)
        x = layers.Activation('relu', name=prefix+'/2/relu')(x)
    convolution = layers.Conv2D(
        fcReLU, (k, k), strides=stride,
        padding='same', use_bias=False, name=prefix+'/2/conv'
    )(x)
    batchnorm = layers.BatchNormalization(
        name=prefix+'/3/bn', center=False, scale=False
    )(convolution)
    negation = layers.Lambda(lambda x: -1*x, name=prefix+'/3/neg')(batchnorm)
    concatenation = layers.Concatenate(name=prefix+'/3/concat')(
        [batchnorm, negation]
    )
    scaleshift = ScaleShift(name=prefix+'3/scale')(concatenation)
    x = layers.Activation('relu', name=prefix+'/3/relu')(scaleshift)
    if fOut is not None:
        x = layers.Conv2D(fOut, 1, name=prefix+'/3/conv')(x)
        if residual:
            if stride != 1 or fOut != xRes.get_shape().as_list()[-1]:
                xRes = layers.Conv2D(fOut, 1, strides=stride, name=prefix+'/proj')(xRes)  # noqa: E501
            x = layers.Add(name=prefix+'/add')([x, xRes])
    return x


def build_backbone(input_shape=(1056, 640, 3)):
    """Build PVANet Backbone. The names of layers are intended to match
    those in
    https://github.com/sanghoon/pva-faster-rcnn/tree/master/models/pvanet
    """
    input_layer = layers.Input(input_shape)
    conv1_1 = crelu(input_layer, 7, 16, 'conv1_1', 2, residual=False)
    pool1_1 = layers.MaxPooling2D(3, 2, name='pool1_1', padding='same')(conv1_1)  # noqa: E501
    conv2_1 = crelu(pool1_1, 3, 24, 'conv2_1', fIn=24, fOut=64, input_bn=False)
    conv2_2 = crelu(conv2_1, 3, 24, 'conv2_2', fIn=24, fOut=64)
    conv2_3 = crelu(conv2_2, 3, 24, 'conv2_3', fIn=24, fOut=64)
    conv3_1 = crelu(conv2_3, 3, 48, 'conv3_1', 2, fIn=48, fOut=128)
    conv3_2 = crelu(conv3_1, 3, 48, 'conv3_2', fIn=48, fOut=128)
    conv3_3 = crelu(conv3_2, 3, 48, 'conv3_3', fIn=48, fOut=128)
    conv3_4 = crelu(conv3_3, 3, 48, 'conv3_4', fIn=48, fOut=128)
    conv4_1 = inception(conv3_4, 64, (48, 128), (24, 48, 48), 256, 'conv4_1', 128, 2)  # noqa: E501
    conv4_2 = inception(conv4_1, 64, (64, 128), (24, 48, 48), 256, 'conv4_2')
    conv4_3 = inception(conv4_2, 64, (64, 128), (24, 48, 48), 256, 'conv4_3')
    conv4_4 = inception(conv4_3, 64, (64, 128), (24, 48, 48), 256, 'conv4_4')
    conv5_1 = inception(conv4_4, 64, (96, 192), (32, 64, 64), 384, 'conv5_1', 128, 2)  # noqa: E501
    conv5_2 = inception(conv5_1, 64, (96, 192), (32, 64, 64), 384, 'conv5_2')
    conv5_3 = inception(conv5_2, 64, (96, 192), (32, 64, 64), 384, 'conv5_3')
    conv5_4 = inception(conv5_3, 64, (96, 192), (32, 64, 64), 384, 'conv5_4', outputBn=True)  # noqa: E501
    downscale = layers.MaxPooling2D(3, strides=2, name='downscale', padding='same')(conv3_4)  # noqa: E501
    upscale = layers.UpSampling2D(size=2, data_format='channels_last', interpolation='bilinear', name='upscale')(conv5_4)  # noqa: E501
    concat = layers.Concatenate(name='concat')([downscale, conv4_4, upscale])
    convf = layers.Conv2D(512, 1, name='convf')(concat)
    model = models.Model(inputs=input_layer, outputs=convf)
    return model


def build_model(backbone):
    """Build EAST Text Detection model on top of ResNet backbone.

    Args:
        backbone: The PVANet model built using build_backbone

    Returns:
        Two models, described below.

        The first is the EAST model with 2 outputs. All outputs
        bounded between 0 and 1 (sigmoid activation).

        Output #1 has shape (B, H // 4, W // 4, 1). It is
            is objectness score for each 4x4 box in the input image.
        Output #2 has shape (B, H // 4, W // 4, 5). It is
            the y1 (top), x2 (right), y2 (bottom), x1 (left), and angle
            offset for each predicted box from the input.

        The second model is the backbone for the EAST model (i.e.
        ResNet 50).
    """
    input_layer = backbone.input
    endpoint_names = reversed([
        'conv3_1/1/relu',
        'conv4_1/incep/relu',
        'conv5_1/incep/relu',
        'conv5_4/last_relu'
    ])
    endpoints = [backbone.get_layer(name) for name in endpoint_names]
    num_outputs = [None, 128, 64, 32]
    x = None
    for i, (filters, endpoint) in enumerate(zip(num_outputs, endpoints)):  # noqa: E501
        if filters is not None:
            x = layers.Concatenate()([x, endpoint.output])
            x = layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)  # noqa: E501
            x = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)  # noqa: E501
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)  # noqa: E501
            x = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)  # noqa: E501
            x = layers.Activation('relu')(x)
        else:
            x = endpoint.output
        if i < 3:
            x = layers.UpSampling2D(
                size=(2, 2),
                interpolation='bilinear',
                name='resize_{0}'.format(i+1)
            )(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)  # noqa: E501
    x = layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = layers.Activation('relu')(x)

    pred_score_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred_score_map')(x)  # noqa: E501
    rbox_geo_map = layers.Conv2D(4, (1, 1), activation='sigmoid', name='rbox_geo_map')(x)  # noqa: E501
    angle_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='rbox_angle_map')(x)  # noqa: E501
    angle_map = layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
    pred_geo_map = layers.Concatenate(axis=3, name='pred_geo_map')([rbox_geo_map, angle_map])  # noqa: E501
    outputs = layers.Concatenate(axis=3)([pred_score_map, pred_geo_map])
    return models.Model(inputs=[input_layer], outputs=outputs)


def loss(y_true, y_pred):
    # Compute score loss
    st, sp = [y[:, :, :, 0] for y in [y_true, y_pred]]  # noqa: E501
    beta = 1 - K.mean(st)
    score_loss = -(beta*st*K.log(sp + K.epsilon()) + (1-beta)*(1-st)*K.log(1. - sp + K.epsilon()))  # noqa: E501

    # Compute geo loss
    d1p, d2p, d3p, d4p = [y_pred[:, :, :, i:i+1] for i in range(1, 5)]
    d1t, d2t, d3t, d4t = [y_true[:, :, :, i:i+1] for i in range(1, 5)]
    thetap, thetat = [y[:, :, :, 5] for y in [y_pred, y_true]]
    wi = K.min(K.concatenate([d2p, d2t], axis=3), axis=3) + K.min(K.concatenate([d4p, d4t], axis=3), axis=3)  # noqa: E501
    hi = K.min(K.concatenate([d1p, d1t], axis=3), axis=3) + K.min(K.concatenate([d3p, d3t], axis=3), axis=3)  # noqa: E501
    intersection = wi*hi
    union = ((d2p + d4p)*(d1p + d3p) + (d2t + d4t)*(d1t + d3t))[:, :, :, 0] - intersection + K.epsilon()  # noqa: E501
    laabb = K.switch(K.equal(st, 1), -K.log((intersection / union) + K.epsilon()), K.zeros_like(intersection))  # noqa: E501
    ltheta = K.switch(K.equal(st, 1), K.abs(1 - K.cos(thetap - thetat)), K.zeros_like(thetap))  # noqa: E501
    geo_loss = laabb + 10*ltheta
    return score_loss + 1*geo_loss


class EAST(Detector):
    """A detector wrapping the
    `EAST Text Detector <https://arxiv.org/abs/1704.03155>`_. This version
    uses an implementation of PVANET for feature extraction,
    as in the original paper.

    Args:
        annotation_config: The annotation configuration to use for detection.
            It must contain `text` as one of the categories.
        input_shape: Tuple of (height, width, n_channels)
        pretrained_backbone: Whether to use a pretrained backbone for the model
        pretrained_top: Whether to use the pretrained full model.
        text_category: The name of the category to use for labeling text.

    Attributes:
        model: The underlying model
        backbone: The backbone for the model (i.e., ResNet50)
    """
    def __init__(
        self,
        annotation_config: core.AnnotationConfiguration=None,
        input_shape: Tuple[int, int]=(None, None, 3),
        pretrained_backbone: bool=False,
        pretrained_top: bool=False,
        text_category='text'
    ):
        if annotation_config is None:
            annotation_config = core.AnnotationConfiguration(['text'])
        assert text_category in annotation_config, \
            'This model only detects text, so, at a minimum, the provided `text_category` must be in annotation_config'  # noqa; E501
        assert len(input_shape) == 3, \
            'Input shape must have shape (h, w, n_channels).'
        if pretrained_top and pretrained_backbone:
            log.warning(
                'pretrained_top makes pretrained_backbone '
                'redundant. Disabling pretrained_backbone.'
            )
            pretrained_backbone = False
        self.annotation_config = annotation_config
        self.text_category = self.annotation_config[text_category]
        self.backbone = build_backbone(input_shape)

        self.model = build_model(self.backbone)
        if pretrained_top or pretrained_backbone:
            raise NotImplementedError('Pretrained weights not yet available.')
        self.compile()

    def invert_targets(
        self,
        y,
        images,
        threshold=0.8,
        **nms_kwargs
    ):
        scenes = []
        for yi, image in zip(y, images):
            scores = yi[:, :, 0:1]
            geo = yi[:, :, 1:]
            # (N, 2): List of coordinates above threshold in
            # the output resolution. The values are (xOut, yOut)
            # and the list is sorted in order of ascending yOut.
            xy_out = np.argwhere(scores[:, :, 0] > threshold)[:, ::-1]

            N = len(xy_out)

            # (N, 2): List of coordinates above threshold
            # in the input resolution. We need only multiply
            # xy_out by 4 to recover the original because
            # the model output map resolution is 1 / 4 of
            # input resolution). The values are (xIn, yIn)
            # and the list is sorted in order of ascending yIn.
            origin = xy_out*4

            # (N, 5): List of adjustments to the raw boxes
            # where the values are (d_top, d_right,
            # d_bottom, d_left, theta)
            geometry = geo[xy_out[:, 1], xy_out[:, 0], :]
            geometry[:, 0] *= image.shape[0]
            geometry[:, 1] *= image.shape[1]
            geometry[:, 2] *= image.shape[0]
            geometry[:, 3] *= image.shape[1]

            boxes = np.zeros((N, 9))
            boxes[:, 8] = scores[scores[:, :, 0] > threshold, 0]
            if len(boxes) == 0:
                annotations = []
            else:
                for i, (xc, yc), (dt, dr, db, dl, theta) in zip(
                    range(N),
                    origin,
                    geometry
                ):
                    theta *= 180/np.pi
                    # Assume x1, y1
                    X = np.array([
                        [0, -dt - db],        # p1 (bottom right)
                        [dr + dl, -dt - db],  # p2 (top right)
                        [dr + dl, 0],         # p3 (top left)
                        [0, 0],               # p4 (bottom left)
                        [dl, -db]             # p5 (anchor offset)
                    ])
                    # According to paper, rotation is about the
                    # bottom left corner, corresponding to x1, y2.
                    M = cv2.getRotationMatrix2D(
                        center=(0, 0),
                        angle=theta,
                        scale=1
                    )
                    X = cv2.transform(np.array([X]), M)[0]
                    X[4, 0] = xc - X[4, 0]
                    X[4, 1] = yc - X[4, 1]
                    X[:4, :] += X[4]
                    boxes[i, :8] = X[:4].reshape(-1)
                nmsIdx = nms.polygons(
                    polys=boxes[:, :8].reshape(-1, 4, 2),
                    scores=boxes[:, 8],
                    **nms_kwargs
                )
                boxes = boxes[nmsIdx]
                annotations = [
                    core.Annotation(
                        category=self.text_category,
                        selection=core.Selection(points=box[:8].reshape((4, 2))),  # noqa: E501
                        score=box[-1]
                    ) for box in boxes
                ]
            scenes.append(
                core.Scene(
                    annotation_config=self.annotation_config,
                    annotations=annotations,
                    image=image
                )
            )
        return scenes

    def compute_targets(
        self,
        collection: core.SceneCollection
    ):
        # y has shape (B, H // 4, W // 4, 6)
        # The entries are score, dt, dr, db, dl, theta
        ys = []
        for scene in collection:
            image_h = scene.image.height
            image_w = scene.image.width
            cy = np.arange(start=0, stop=image_h, dtype='float32', step=4)
            cx = np.arange(start=0, stop=image_w, dtype='float32', step=4)
            xy = np.concatenate([
                np.tile(cx, len(cy))[:, np.newaxis],
                np.repeat(cy, len(cx))[:, np.newaxis]
            ], axis=1)
            scores = np.zeros((len(xy), 1))
            geometry = np.zeros((len(xy), 5))

            for ann in scene.annotations:
                if ann.category != self.text_category:
                    continue
                ((xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl)), theta = ann.selection.rbox()  # noqa; E501
                positive = ann.selection.contains_points(xy)
                origin = xy[positive]

                # Horizontal distance
                lh = np.sqrt((ybr - ybl) ** 2 + (xbr - xbl) ** 2)

                # Vertical distance
                lv = np.sqrt((ytl - ybl) ** 2 + (xtl - xbl) ** 2)

                dt = abs(
                    (ytr - ytl)*origin[:, 0] -
                    (xtr - xtl)*origin[:, 1] +
                    xtr*ytl - ytr*xtl
                ) / lh
                db = lv - dt

                dr = abs(
                    (ytr - ybr)*origin[:, 0] -
                    (xtr - xbr)*origin[:, 1] +
                    xtr*ybr - ytr*xbr
                ) / lv
                dl = lh - dr
                scores[positive] = 1

                geometry[positive, 0] = dt / image_h  # dt
                geometry[positive, 1] = dr / image_w  # dr
                geometry[positive, 2] = db / image_h  # db
                geometry[positive, 3] = dl / image_w  # dl
                geometry[positive, 4] = theta

            dt = geometry[:, 0]
            dl = geometry[:, 1]
            lv = geometry[:, 0] + geometry[:, 2]
            lh = geometry[:, 1] + geometry[:, 3]
            dlr = np.divide(dl, lh, out=np.zeros_like(dl), where=lh != 0)
            dtr = np.divide(dt, lv, out=np.zeros_like(dt), where=lv != 0)

            # Only include shrunken boxes
            margin = 0.15
            exclude = (
                (dlr <= margin) | (dlr >= 1-margin) |
                (dtr <= margin) | (dtr >= 1-margin)
            ) & (scores[:, 0] == 1)
            scores[exclude] = 0

            y1i = scores.reshape((len(cy), len(cx), 1))
            y2i = geometry.reshape((len(cy), len(cx), 5))
            ys.append(np.concatenate([y1i, y2i], axis=2))
        return np.float32(ys)

    def compute_inputs(self, images: List[core.Image]):
        assert all(
            [all(s % 32 == 0 for s in image.shape[:2]) for image in images]
        ), \
            'Input shapes must all be multiples of 32.'
        assert all(
            image.shape[0] == image.shape[1] for image in images
        ), 'All images must be square.'
        return np.float32([
            image.scaled(-1, 1)
            for image in images
        ])

    def compile(self):
        self.model.compile(
            loss=loss,
            optimizer=optimizers.Adam()
        )
