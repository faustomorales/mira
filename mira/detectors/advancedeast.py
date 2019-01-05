from typing import Tuple
from os import path
import logging
import itertools

from keras import applications, layers, models, optimizers
from keras import backend as K
import numpy as np
import cv2

from ..core import (
    Scene,
    Selection,
    Annotation,
    AnnotationConfiguration
)
from .detector import Detector
from .. import utils

log = logging.getLogger(__name__)


def build_backbone(input_shape=(None, None, 3), pretrained=False):
    input_layer = layers.Input(input_shape)
    vgg16 = applications.vgg16.VGG16(
        input_tensor=input_layer,
        weights='imagenet' if pretrained else None,
        include_top=False
    )
    features = [
        vgg16.get_layer('block{0}_pool'.format(i)).output for i in range(2, 6)
    ]
    return models.Model(inputs=input_layer, outputs=features)


def build_model(backbone):
    x = None
    for outputIdx in range(len(backbone.outputs)-1, 0, -1):
        hIdx = len(backbone.outputs) - outputIdx + 1
        filters = 2**(outputIdx + 4)
        if x is None:
            x = backbone.outputs[outputIdx]
        x = layers.UpSampling2D((2, 2), name='h{0}/upsample'.format(hIdx))(x)
        x = layers.Concatenate(name='h{0}/concat'.format(hIdx))([x, backbone.outputs[outputIdx - 1]])  # noqa: E501
        x = layers.BatchNormalization(name='h{0}/bn1'.format(hIdx))(x)
        x = layers.Conv2D(
            filters=filters, kernel_size=1, activation='relu', padding='same', name='h{0}/conv1'.format(hIdx)  # noqa: E501
        )(x)
        x = layers.BatchNormalization(name='h{0}/bn2'.format(hIdx))(x)
        x = layers.Conv2D(
            filters=filters, kernel_size=3, activation='relu', padding='same', name='h{0}/conv2'.format(hIdx)  # noqa: E501
        )(x)
    x = layers.BatchNormalization(name='features/bn')(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='features/conv')(x)  # noqa: E501
    y1 = layers.Conv2D(filters=1, kernel_size=1, padding='same', name='inside_score', activation='sigmoid')(x)  # noqa: E501
    y2 = layers.Conv2D(filters=2, kernel_size=1, padding='same', name='side_vertex_code', activation='sigmoid')(x)  # noqa: E501
    y3 = layers.Conv2D(filters=4, kernel_size=1, padding='same', name='side_vertex_coord')(x)  # noqa: E501
    east_detect = layers.Concatenate(axis=-1, name='east_detect')([y1, y2, y3])
    return models.Model(inputs=backbone.inputs, outputs=east_detect, name='advanced_east')  # noqa: E501


def loss(y_true, y_pred):
    eps = K.epsilon()

    # Compute score loss
    st, sp = y_true[..., 0:1], y_pred[..., 0:1]
    beta = 1 - K.mean(st)
    score_loss = -(beta*st*K.log(sp + eps) + (1-beta)*(1-st)*K.log(1. - sp + eps))  # noqa: E501
    score_loss = K.mean(score_loss)

    # Compute end loss
    et, ep = y_true[..., 1:3], y_pred[..., 1:3]
    beta = 1 - (K.mean(et[..., 0]) / (K.mean(st) + eps))
    end_loss = -(beta*et*K.log(ep + eps) + (1-beta)*(1-et)*K.log(1. - ep + eps))  # noqa: E501
    end_loss = K.sum(end_loss * st) / (K.sum(st) + eps)

    # Compute vertex loss
    vt, vp = y_true[..., 3:], y_pred[..., 3:]
    sizes = 4*K.sqrt(K.sum(K.square(vt[..., 0:2] - vt[..., 2:4]), axis=-1)) + eps  # noqa: E501
    diff = K.abs(vp - vt)
    smooth_l1 = (K.sum(K.switch(K.less(diff, 1), 0.5 * K.square(diff), diff - 0.5), axis=-1) / sizes) * et[..., 0]  # noqa: E501
    geo_loss = K.sum(smooth_l1) / (K.sum(st) + eps)

    score_weight, end_weight, geo_weight = 4.0, 1, 1
    return score_weight*score_loss + end_weight*end_loss + geo_weight*geo_loss


class AdvancedEAST(Detector):
    """A detector wrapping the
    `Advanced EAST Text Detector <https://github.com/huoyijie/AdvancedEAST>`_.
    The original code was ported over. The pretrained weights also originate
    from the linked repository. This version of EAST is expected to perform
    better on longer text boxes relative to the input image.

    Args:
        annotation_config: The annotation configuration to use for detection.
            It must contain `text` as one of the categories.
        input_shape: Tuple of (height, width, n_channels)
        pretrained_backbone: Whether to use a pretrained backbone for the model
        pretrained_top: Whether to use the pretrained full model.
        text_category: The name of the category to use for labeling text.

    Attributes:
        model: The underlying model
        backbone: The backbone for the model (i.e., VGG16)
    """
    def __init__(
        self,
        annotation_config: AnnotationConfiguration=None,
        input_shape: Tuple[int, int]=(None, None, 3),
        pretrained_backbone: bool=True,
        pretrained_top: bool=False,
        text_category='text'
    ):
        if annotation_config is None:
            annotation_config = AnnotationConfiguration(['text'])
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
        self.backbone = build_backbone(
            input_shape=input_shape,
            pretrained=pretrained_backbone
        )
        self.model = build_model(self.backbone)
        if pretrained_top:
            weights_path = utils.get_file(
                origin='https://storage.googleapis.com/miradata/weights/advancedeast/east_model_weights_3T736.h5',  # noqa; E501
                file_hash='1a5ef4e304cd71358fcfd4403859c38735738eed2961ea9b34484c7e4199d4f6',  # noqa; E501
                cache_subdir=path.join('weights', 'advancedeast'),
                hash_algorithm='sha256',
                extract=False
            )
            self.model.load_weights(weights_path)
        self.compile()

    def invert_targets(self, y, images, threshold=0.9):
        scenes = []
        placeholder = np.zeros(y.shape[1:-1])

        def contourCoords(contour):
            placeholder[...] = 0
            cv2.drawContours(
                image=placeholder,
                contours=[contour],
                contourIdx=-1,
                color=1,
                thickness=-1
            )
            yp, xp = np.where(placeholder > 0)
            return xp, yp

        def vertices_from_mask(mask, predictions):
            contours, _ = cv2.findContours(
                image=mask,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            vertices = []
            for contour in contours:
                placeholder[...] = 0
                xc, yc = contourCoords(contour)
                w, t, dx1, dy1, dx2, dy2 = predictions[yc, xc, 1:].T
                w /= w.sum()
                x1, y1 = 4*xc + dx1, 4*yc + dy1
                x2, y2 = 4*xc + dx2, 4*yc + dy2
                x1, y1, x2, y2 = map(
                    lambda v: sum(w * v),
                    [x1, y1, x2, y2]
                )
                vertices.append([
                    [x1, y1],
                    [x2, y2]
                ])
            return vertices

        for yi, image in zip(y, images):
            mask = np.uint8(((yi[..., :3] > threshold) & (yi[..., 0:1] > threshold))*255)  # noqa: E501
            regionContours, _ = cv2.findContours(
                image=mask[..., 0],
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for regionContour in regionContours:
                xc, yc = contourCoords(regionContour)
                mVT = mask[..., 1:3]*placeholder[..., np.newaxis]
                mH = np.uint8(255*((mVT[..., 0] > 127.5) & (mVT[..., 1] < 127.5)))  # noqa: E501
                mT = np.uint8(255*((mVT[..., 0] > 127.5) & (mVT[..., 1] > 127.5)))  # noqa: E501
                if mH.max() == 0 or mT.max() == 0:
                    continue
                hVertices = vertices_from_mask(mH, yi)
                tVertices = vertices_from_mask(mT, yi)
                subcontours = [
                    (np.array(h + t).reshape(4, 1, 2) // 4).astype('int32') for h, t in itertools.product(hVertices, tVertices)  # noqa: E501
                ]
                areas = [
                    cv2.contourArea(c) for c in subcontours
                ]
                subcontour = subcontours[np.argmax(areas)]
                xs, ys = contourCoords(np.int32(subcontour))
                score = yi[ys, xs, 0].mean()
                boxes.append(
                    subcontour.reshape(-1).tolist() + [score]
                )
            boxes = np.float32(boxes)
            annotations = [
                Annotation(
                    selection=Selection(box[:8].reshape(4, 2)),
                    score=box[-1],
                    category=self.text_category
                ).resize(4) for box in boxes
            ]
            scenes.append(
                Scene(
                    image=image,
                    annotation_config=self.annotation_config,
                    annotations=annotations
                )
            )
        return scenes

    def compute_targets(self, collection):
        # y has shape (B, H // 4, W // 4, 7)
        # The entries are score, left, right, dx1, dy1, dx2, dy2
        B = len(collection)
        H = collection[0].image.height
        W = collection[0].image.width
        y = np.zeros((B, H // 4, W // 4, 7))
        points = np.mgrid[:y.shape[1], :y.shape[2]].reshape(2, -1).T
        for i, scene in enumerate(collection):
            for ann in scene.annotations:
                if ann.category != self.text_category:
                    continue
                selection = ann.selection.resize(0.25)
                xa, ya = points[selection.contains_points(points)].T
                ((xtl, ytl), (xtr, ytr), (xbr, ybr), (xbl, ybl)), theta = selection.rbox()  # noqa: E501

                # Horizontal distance
                lx = np.sqrt((ybr - ybl) ** 2 + (xbr - xbl) ** 2)

                # Vertical distance
                ly = np.sqrt((ytl - ybl) ** 2 + (xtl - xbl) ** 2)

                # Distance to top edge
                dy = abs(
                    (ytr - ytl)*xa -
                    (xtr - xtl)*ya +
                    xtr*ytl - ytr*xtl
                ) / lx

                # Distance to left edge
                dx = lx - abs(
                    (ytr - ybr)*xa -
                    (xtr - xbr)*ya +
                    xtr*ybr - ytr*xbr
                ) / ly

                # Scale distances between 0 and 1.
                # dx == 0 corresponds with the left edge
                # and dx == 1 corresponds with the right edge.
                dx /= lx
                dy /= ly

                # Positive scores are for the box with the radius
                # from 0 to 0.8. This corresponds to shrinking
                # the overall box by 0.1. So 0.1 -> 0.9 on the
                # overall box are positive.
                isPositive = (dx > 0.1) & (dx < 0.9) & (dy > 0.1) & (dy < 0.9)  # noqa: E501

                edgeMargin = 0.2

                # Head and tail are defined by longest edge.
                if ly > lx:
                    dLong = dy
                    xh1, yh1 = xtl, ytl
                    xh2, yh2 = xtr, ytr
                    xt1, yt1 = xbl, ybl
                    xt2, yt2 = xbr, ybr
                else:
                    dLong = dx
                    xh1, yh1 = xtl, ytl
                    xh2, yh2 = xbl, ybl
                    xt1, yt1 = xtr, ytr
                    xt2, yt2 = xbr, ybr

                isBorder = ((dLong < edgeMargin) | (dLong > 1 - edgeMargin)) & isPositive  # noqa: E501
                isTail = (dLong > (1 - edgeMargin)) & isPositive
                isHead = (dLong < edgeMargin) & isPositive

                y[i, ya[isPositive], xa[isPositive], 0] = 1
                y[i, ya[isBorder], xa[isBorder], 1] = 1
                y[i, ya[isTail], xa[isTail], 2] = 1
                y[i, ya[isHead], xa[isHead], 3:7] = np.concatenate([
                    v[:, np.newaxis] for v in [(xh1 - xa), (yh1 - ya), (xh2 - xa), (yh2 - ya)]  # noqa: E501
                ], axis=1)[isHead]*4
                y[i, ya[isTail], xa[isTail], 3:7] = np.concatenate([
                    v[:, np.newaxis] for v in [(xt1 - xa), (yt1 - ya), (xt2 - xa), (yt2 - ya)]  # noqa: E501
                ], axis=1)[isTail]*4
            return y

    def compute_inputs(self, images):
        assert all(
            [all(s % 32 == 0 for s in image.shape[:2]) for image in images]
        ), \
            'Input shapes must all be multiples of 32.'
        assert all(
            image.shape[0] == image.shape[1] for image in images
        ), 'All images must be square.'
        return applications.vgg16.preprocess_input(
            np.float32(images),
            mode='tf'
        )

    def compile(self):
        self.model.compile(
            loss=loss,
            optimizer=optimizers.Adam()
        )