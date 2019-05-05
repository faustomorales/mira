import math
import logging
from os import path
from typing import List, Union

from keras import layers, models
import numpy as np
import cv2

from .. import core, utils
from .detector import Detector

log = logging.getLogger(__name__)

""" MTCNN
This intends to replicate the MTCNN architecture that comes with FaceNet. It
has one small change from the original (see detect function). To compare
outputs, you can use something like the following.

from mira import core, detectors
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

url = 'https://www.abc.net.au/news/image/9499864-3x2-700x467.jpg'
image = core.Image.read(url)

detector1 = detectors.MTCNN()
detector2 = MTCNN()

faces1 = detector1.detect(image)
faces2 = detector2.detect_faces(image)
scene1 = core.Scene(
    annotations=faces1,
    image=image,
    annotation_config=detector1.annotation_config
)
scene2 = core.Scene(
    annotation_config=detector1.annotation_config,
    annotations=[core.Annotation(
        category=detector1.annotation_config['face'],
        selection=core.Selection([
            [f['box'][0], f['box'][1]],
            [f['box'][0] + f['box'][2], f['box'][1]+f['box'][3]]
        ])
    ) for f in faces2],
    image=image
)
print('Our boxes:', [f.selection.xywh() for f in faces1])
print('Their boxes:', [f['box'] for f in faces2])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
scene1.show(ax1)
scene2.show(ax2)
"""

WEIGHTS_CONFIG = {
    "pnet": {
        "hash":
        "3285cf7a3de2651c5784cb9e32013f5919aae95fd1ed1bc371dd9691affb39af",  # noqa: E501
        "url":
        "https://storage.googleapis.com/miradata/weights/mtcnn/det1.npy",
    },
    "rnet": {
        "hash":
        "716b8b83e42476791c9096f14dbb09fefc88bf5c7ec876b1683f9acd52b3f39c",  # noqa: E501
        "url":
        "https://storage.googleapis.com/miradata/weights/mtcnn/det2.npy",
    },
    "onet": {
        "hash":
        "396ead803d85d3443307ff8f45fb6aed2536579b415a4f4d4cb8f93ea6b1476a",  # noqa: E501
        "url":
        "https://storage.googleapis.com/miradata/weights/mtcnn/det3.npy",
    },
}


def make_pnet(input_shape=(None, None, 3)):
    x = layers.Input(input_shape)
    conv1 = layers.Conv2D(kernel_size=(3, 3),
                          filters=10,
                          strides=(1, 1),
                          padding="valid",
                          name="conv1")(x)
    relu1 = layers.PReLU(name="PReLU1", shared_axes=[1, 2])(conv1)
    mp1 = layers.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2),
                              padding="same",
                              name="pool1")(relu1)
    conv2 = layers.Conv2D(kernel_size=(3, 3),
                          filters=16,
                          strides=(1, 1),
                          padding="valid",
                          name="conv2")(mp1)
    relu2 = layers.PReLU(name="PReLU2", shared_axes=[1, 2])(conv2)
    conv3 = layers.Conv2D(kernel_size=(3, 3),
                          filters=32,
                          strides=(1, 1),
                          padding="valid",
                          name="conv3")(relu2)
    relu3 = layers.PReLU(name="PReLU3", shared_axes=[1, 2])(conv3)
    conv4_1 = layers.Conv2D(kernel_size=(1, 1),
                            filters=2,
                            strides=(1, 1),
                            padding="same",
                            name="conv4-1")(relu3)
    prob1 = layers.Softmax(axis=3, name="prob1")(conv4_1)
    conv4_2 = layers.Conv2D(kernel_size=(1, 1),
                            filters=4,
                            strides=(1, 1),
                            padding='same',
                            name="conv4-2")(relu3)
    return models.Model(inputs=x, outputs=[prob1, conv4_2])


def make_rnet(input_shape=(24, 24, 3)):
    x = layers.Input(input_shape)
    conv1 = layers.Conv2D(kernel_size=(3, 3),
                          filters=28,
                          strides=(1, 1),
                          padding="valid",
                          name="conv1")(x)
    relu1 = layers.PReLU(name="prelu1", shared_axes=[1, 2])(conv1)
    mp1 = layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              name="pool1")(relu1)
    conv2 = layers.Conv2D(kernel_size=(3, 3),
                          filters=48,
                          strides=(1, 1),
                          padding="valid",
                          name="conv2")(mp1)
    relu2 = layers.PReLU(name="prelu2", shared_axes=[1, 2])(conv2)
    mp2 = layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="valid",
                              name="pool2")(relu2)
    conv3 = layers.Conv2D(kernel_size=(2, 2),
                          filters=64,
                          strides=(1, 1),
                          padding="valid",
                          name="conv3")(mp2)
    relu3 = layers.PReLU(name="prelu3", shared_axes=[1, 2])(conv3)
    conv4 = layers.Conv2D(kernel_size=(3, 3),
                          filters=128,
                          strides=(1, 1),
                          padding="valid",
                          name="conv4")(relu3)
    relu4 = layers.PReLU(name="prelu4", shared_axes=[1, 2])(conv4)
    conv5_1 = layers.Conv2D(kernel_size=(1, 1),
                            filters=2,
                            strides=(1, 1),
                            padding="valid",
                            name="conv5-1")(relu4)
    prob1 = layers.Softmax(axis=3)(conv5_1)
    conv5_2 = layers.Conv2D(kernel_size=(1, 1),
                            filters=4,
                            strides=(1, 1),
                            padding="valid",
                            name="conv5-2")(relu4)
    clf = layers.Reshape((2, ), name='clf')(prob1)
    reg = layers.Reshape((4, ), name='reg')(conv5_2)
    return models.Model(inputs=x, outputs=[clf, reg])


def make_onet(input_shape=(48, 48, 3)):
    x = layers.Input(input_shape)
    conv1 = layers.Conv2D(kernel_size=(3, 3),
                          filters=32,
                          strides=(1, 1),
                          padding="valid",
                          name="conv1")(x)
    relu1 = layers.PReLU(name="prelu1", shared_axes=[1, 2])(conv1)
    mp1 = layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="same",
                              name="pool1")(relu1)
    conv2 = layers.Conv2D(kernel_size=(3, 3),
                          filters=64,
                          strides=(1, 1),
                          padding="valid",
                          name="conv2")(mp1)
    relu2 = layers.PReLU(name="prelu2", shared_axes=[1, 2])(conv2)
    mp2 = layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="valid",
                              name="pool2")(relu2)
    conv3 = layers.Conv2D(kernel_size=(3, 3),
                          filters=64,
                          strides=(1, 1),
                          padding="valid",
                          name="conv3")(mp2)
    relu3 = layers.PReLU(name="prelu3", shared_axes=[1, 2])(conv3)
    mp3 = layers.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2),
                              padding="same",
                              name="pool3")(relu3)
    conv4 = layers.Conv2D(kernel_size=(2, 2),
                          filters=128,
                          strides=(1, 1),
                          padding="valid",
                          name="conv4")(mp3)
    relu4 = layers.PReLU(name="prelu4", shared_axes=[1, 2])(conv4)
    conv5 = layers.Conv2D(kernel_size=(3, 3),
                          filters=256,
                          padding="valid",
                          name="conv5")(relu4)
    relu5 = layers.PReLU(name="prelu5", shared_axes=[1, 2])(conv5)
    conv6_1 = layers.Conv2D(kernel_size=(1, 1), filters=2,
                            name="conv6-1")(relu5)
    prob1 = layers.Softmax(axis=3, name="prob1")(conv6_1)
    conv6_2 = layers.Conv2D(kernel_size=(1, 1), filters=4,
                            name="conv6-2")(relu5)
    conv6_3 = layers.Conv2D(kernel_size=(1, 1), filters=10,
                            name="conv6-3")(relu5)
    clf = layers.Reshape((2, ), name='clf')(prob1)
    reg = layers.Reshape((4, ), name='reg')(conv6_2)
    feats = layers.Reshape((10, ), name='feats')(conv6_3)
    return models.Model(inputs=x, outputs=[clf, reg, feats])


def load_weights(network, weights_file):
    for layer_name, weights in np.load(
        weights_file,
        encoding="latin1",
        allow_pickle=True
    ).item().items():
        layer = network.get_layer(layer_name)
        if layer_name.lower().startswith("conv"):
            coefs = weights["weights"].reshape(
                layer.weights[0].shape.as_list())
            biases = weights["biases"].reshape(
                layer.weights[1].shape.as_list())
            layer.set_weights(weights=[coefs, biases])
        elif layer_name.lower().startswith("prelu"):
            alpha = weights["alpha"].reshape(layer.weights[0].shape)
            layer.set_weights(weights=[alpha])
        else:
            raise NotImplementedError("Cannot handle layer: " + layer_name)


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 9))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_s = np.argsort(s)

    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while sorted_s.size > 0:
        i = sorted_s[-1]
        pick[counter] = i
        counter += 1
        idx = sorted_s[0:-1]

        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        sorted_s = sorted_s[np.where(o <= threshold)]

    pick = pick[0:counter]

    return boxes[pick]


def padded(x1y1x2y2s):
    x1y1x2y2s = x1y1x2y2s.copy()
    x1y1x2y2s[:, :2] = np.clip(x1y1x2y2s[:, :2], a_min=1, a_max=np.inf) - 1
    return x1y1x2y2s


def square(x1y1x2y2s):
    x1y1x2y2s = x1y1x2y2s.copy()
    h = x1y1x2y2s[:, 3] - x1y1x2y2s[:, 1]
    w = x1y1x2y2s[:, 2] - x1y1x2y2s[:, 0]
    length = np.maximum(w, h)
    x1y1x2y2s[:, 0] = x1y1x2y2s[:, 0] + w * 0.5 - length * 0.5
    x1y1x2y2s[:, 1] = x1y1x2y2s[:, 1] + h * 0.5 - length * 0.5
    x1y1x2y2s[:, 2:4] = x1y1x2y2s[:, 0:2] + length[:, np.newaxis]
    x1y1x2y2s[:, :4] = np.fix(x1y1x2y2s[:, :4])
    return x1y1x2y2s


def regress(boxes):
    boxes = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] += boxes[:, 5] * w
    boxes[:, 1] += boxes[:, 6] * h
    boxes[:, 2] += boxes[:, 7] * w
    boxes[:, 3] += boxes[:, 8] * h
    boxes[:, 5:] = 0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return boxes


def fix_scale(x):
    return (x.astype('float64') - 127.5) * 0.0078125


def run_stage(network, image, x1y1x2y2s, cell_size):
    # we need to use a padded version of x1y1x2y2s, I guess?
    if len(x1y1x2y2s) == 0:
        return np.empty((0, 9))
    crops = [
        image[y1:y2, x1:x2].astype('float64')
        for x1, y1, x2, y2 in padded(x1y1x2y2s[:, :4]).astype('int32')
    ]
    crops = [
        image for image in crops if image.shape[0] > 0 and image.shape[1] > 0
    ]
    crops = [
        cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        for image in crops
    ]
    X = np.float64(crops)
    X = fix_scale(X.transpose(0, 2, 1, 3))
    clf, reg = [v for v in network.predict(X)[:2]]

    x1y1x2y2s[:, 4] = clf[:, 1]
    x1y1x2y2s[:, :4] = np.fix(x1y1x2y2s[:, :4])
    x1y1x2y2s[:, 5:] = reg
    return x1y1x2y2s


class MTCNN(Detector):
    """A detector wrapping MTCNN for face detection. Based on the
    implementation found as part of
    `facenet <https://github.com/davidsandberg/facenet>`_. The only
    difference is we've implemented it using keras and we make one
    tweak to when nms is applied for Stage 2. See comments in
    code for detail. Training is not supported at this time. We also
    do not support recording the locations of facial landmarks. The model
    is documented in `the original paper
    <https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf>`_.

    Example usage:

    .. highlight:: python
    .. code-block:: python

        from mira import core, detectors

        url = 'https://www.abc.net.au/news/image/9499864-3x2-700x467.jpg'
        image = core.Image.read(url)
        detector = detectors.MTCNN()
        faces = detector.detect(image)
        scene = core.Scene(
            annotations=faces,
            image=image,
            annotation_config=detector.annotation_config
        )

    Attributes:
        pnet: The pnet model (stage 1)
        rnet: The rnet model (stage 2)
        onet: The onet model (stage 3)
    """
    def __init__(self):
        self.annotation_config = core.AnnotationConfiguration(["face"])
        self.pnet = make_pnet()
        self.rnet = make_rnet()
        self.onet = make_onet()

        load_weights(
            self.pnet,
            utils.get_file(
                origin=WEIGHTS_CONFIG["pnet"]["url"],  # noqa; E501
                file_hash=WEIGHTS_CONFIG["pnet"]["hash"],  # noqa; E501
                cache_subdir=path.join("weights", "mtcnn"),
                hash_algorithm="sha256",
                extract=False,
            ),
        )
        load_weights(
            self.rnet,
            utils.get_file(
                origin=WEIGHTS_CONFIG["rnet"]["url"],  # noqa; E501
                file_hash=WEIGHTS_CONFIG["rnet"]["hash"],  # noqa; E501
                cache_subdir=path.join("weights", "mtcnn"),
                hash_algorithm="sha256",
                extract=False,
            ),
        )
        load_weights(
            self.onet,
            utils.get_file(
                origin=WEIGHTS_CONFIG["onet"]["url"],  # noqa; E501
                file_hash=WEIGHTS_CONFIG["onet"]["hash"],  # noqa; E501
                cache_subdir=path.join("weights", "mtcnn"),
                hash_algorithm="sha256",
                extract=False,
            ),
        )

    def compute_inputs(self, images):
        raise NotImplementedError('MTCNN is not yet complete.')

    def compute_targets(self, collection):
        raise NotImplementedError('MTCNN is not yet complete.')

    def detect(
        self,
        image: Union[core.Image, np.ndarray],
        minsize=20,
        factor=0.709,
        thresholds=None
    ) -> List[core.Annotation]:
        """Detects faces in an image, and returns bounding boxes and points for them.

        image: input image
        minsize: Minimum face size
        factor: The factor by which to step down scales in the
         image.
        thresholds: The detection thresholds (one for each of three stages)
        """
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        image = image.view(core.Image)
        height, width = image.shape[:2]

        # Scale of MTCNN cell to minimum face size
        m = 12.0 / minsize

        # Reequired image size to have a 12x12
        # region correspond to minimum face
        # size
        minl = min([height, width]) * m
        n_scales = math.ceil(math.log(12 / minl, factor))
        scales = [m * np.power(factor, n) for n in range(n_scales)]

        # Stage 1
        x1y1x2y2s = []
        stride = 2
        cell_size = 12
        for idx, scale in enumerate(scales):
            scaled_image = image.resize(scale, interpolation=cv2.INTER_AREA)
            X = np.float64([
                fix_scale(scaled_image.transpose(1, 0, 2))
            ])
            y = [v.transpose(0, 2, 1, 3) for v in self.pnet.predict(X)]
            clf, reg = y[0][0], y[1][0]
            yc, xc = np.where(clf[..., 1].transpose() >= thresholds[0])
            score = clf[xc, yc, 1:]
            reg = reg[xc, yc]
            bb = np.vstack([yc, xc]).transpose()
            x1y1 = np.fix((stride * bb + 1) / scale)
            x2y2 = np.fix((stride * bb + 1 + cell_size - 1) / scale)
            x1y1x2y2s_current = np.concatenate(
                [x1y1, x2y2, score, reg], axis=1
            )
            x1y1x2y2s.append(nms(x1y1x2y2s_current, 0.5, "Union"))
        x1y1x2y2s = np.concatenate(x1y1x2y2s)
        x1y1x2y2s = nms(x1y1x2y2s, 0.7, "Union")
        x1y1x2y2s = regress(x1y1x2y2s)
        x1y1x2y2s = square(x1y1x2y2s)

        # Stage 2
        x1y1x2y2s = run_stage(self.rnet, image, x1y1x2y2s, 24)
        x1y1x2y2s = x1y1x2y2s[x1y1x2y2s[:, 4] > thresholds[1]]

        # Note the ordering here of applying regression and nms differs from
        # the original implementation. In the original, nms is applied first
        # and regression second on Stage 2. But before regression, the boxes
        # are the same as they were before, we just have a smaller subset of
        # them, making the nms call superfluous. I've swapped the order here
        # to be consistent with Stage 3, where nms is applied after regression
        # regression. To recover the original behavior, the nms and regression
        # lines should be swapped back.
        # https://github.com/davidsandberg/facenet/blob/096ed770f163957c1e56efa7feeb194773920f6e/src/align/detect_face.py#L377
        x1y1x2y2s = regress(x1y1x2y2s)
        x1y1x2y2s = nms(x1y1x2y2s, 0.7, 'Union')
        x1y1x2y2s = square(x1y1x2y2s)

        # Stage 3
        x1y1x2y2s = run_stage(self.onet, image, x1y1x2y2s, 48)
        x1y1x2y2s = x1y1x2y2s[x1y1x2y2s[:, 4] > thresholds[2]]
        x1y1x2y2s = regress(x1y1x2y2s)
        x1y1x2y2s = nms(x1y1x2y2s, 0.7, 'Min')

        return [
            core.Annotation(
                selection=core.Selection(points=[[x1, y1], [x2, y2]]),
                category=self.annotation_config['face'],
                score=s
            ) for x1, y1, x2, y2, s in x1y1x2y2s[:, :5]
        ]
