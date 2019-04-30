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


def make_pnet(input_shape=(12, 12, 3)):
    x = layers.Input(input_shape)
    conv1 = layers.Conv2D(kernel_size=(3, 3),
                          filters=10,
                          strides=(1, 1),
                          padding="valid",
                          name="conv1")(x)
    relu1 = layers.PReLU(name="PReLU1", shared_axes=[1, 2])(conv1)
    mp1 = layers.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2),
                              padding="valid",
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
                            padding="valid",
                            name="conv4-1")(relu3)
    prob1 = layers.Softmax(axis=3, name="prob1")(conv4_1)
    conv4_2 = layers.Conv2D(kernel_size=(1, 1),
                            filters=4,
                            strides=(1, 1),
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
    return models.Model(inputs=x, outputs=[prob1, conv5_2])


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
    return models.Model(inputs=x, outputs=[prob1, conv6_2, conv6_3])


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
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sortIdx = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while sortIdx.size > 0:
        i = sortIdx[-1]
        pick[counter] = i
        counter += 1
        idx = sortIdx[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method == "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        sortIdx = sortIdx[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def calculate_boxes(clf,
                    reg,
                    scale,
                    threshold,
                    cell_size,
                    stride=2,
                    offset=(0, 0)):
    """Use heatmap to generate bounding boxes"""
    y, x = np.where(clf[..., 1].transpose() >= threshold)
    score = clf[x, y, 1:]
    reg = reg[x, y]
    bb = np.vstack([y, x]).transpose()
    x1y1 = np.fix((stride * bb + 1) / scale) + offset
    x2y2 = np.fix((stride * bb + 1 + cell_size - 1) / scale) + offset
    x1y1x2y2sd = np.concatenate([x1y1, x2y2, score, reg], axis=1)
    return x1y1x2y2sd


def pad_and_square(x1y1x2y2s, width, height, pad=0.1):
    xpad = pad * (x1y1x2y2s[:, 2] - x1y1x2y2s[:, 0])
    ypad = pad * (x1y1x2y2s[:, 3] - x1y1x2y2s[:, 1])
    x1y1x2y2s[:, 0] -= xpad
    x1y1x2y2s[:, 2] += xpad
    x1y1x2y2s[:, 1] -= ypad
    x1y1x2y2s[:, 3] += ypad
    x1y1x2y2s[:, 0:4:2] = x1y1x2y2s[:, 0:4:2].clip(0, width)
    x1y1x2y2s[:, 1:4:2] = x1y1x2y2s[:, 1:4:2].clip(0, height)
    length = (x1y1x2y2s[:, 2:4] - x1y1x2y2s[:, :2]).max(axis=1)
    x1y1x2y2s[:, 2:4] = x1y1x2y2s[:, :2] + length[:, np.newaxis]


def apply_regression(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] += boxes[:, 5] * w
    boxes[:, 1] += boxes[:, 6] * h
    boxes[:, 2] += boxes[:, 7] * w
    boxes[:, 3] += boxes[:, 8] * h
    boxes[:, 5:] = 0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]


def run_stage(network, image, x1y1x2y2s, cell_size, threshold):
    crops = [
        image[y1:y2, x1:x2]
        for x1, y1, x2, y2 in x1y1x2y2s[:, :4].round(0).astype("int32")
    ]
    crops = [
        image for image in crops if image.shape[0] > 0 and image.shape[1] > 0
    ]
    X, scales = zip(*[image.fit(cell_size, cell_size) for image in crops])
    X = np.float32([x.transpose(1, 0, 2).scaled(-1, 1) for x in X])
    y = [v.transpose(0, 2, 1, 3) for v in network.predict(X)]
    x1y1x2y2s = np.concatenate([
        calculate_boxes(
            clf,
            box,
            scale=scale,
            threshold=threshold,
            offset=(current[0], current[1]),
            cell_size=cell_size,
        ) for clf, box, scale, current in zip(*y[:2], scales, x1y1x2y2s)
    ])
    apply_regression(x1y1x2y2s)
    return x1y1x2y2s


class MTCNN(Detector):
    """A detector wrapping MTCNN for face detection. Based on the
    implementation found as part of
    `facenet <https://github.com/davidsandberg/facenet>`_. The only
    difference is we've implemented it using keras. Training is not
    supported at this time. The model is documented in `the original
    paper
    <https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf>`_.

    Attributes:
        pnet: The pnet model (stage 1)
        rnet: The rnet model (stage 2)
        onet: The onet model (stage 3)
    """
    def __init__(self):
        input_shape = (None, None, 3)
        self.annotation_config = core.AnnotationConfiguration(["face"])
        self.pnet = make_pnet(input_shape=input_shape)
        self.rnet = make_rnet(input_shape=input_shape)
        self.onet = make_onet(input_shape=input_shape)

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
        threshold=0.8
    ) -> List[core.Annotation]:
        """Detects faces in an image, and returns bounding boxes and points for them.

        image: input image
        minsize: Minimum face size
        factor: The factor by which to step down scales in the
         image.
        threshold: The detection threshold.
        """
        image = image.view(core.Image)
        height, width = image.shape[:2]
        m = 12.0 / minsize  # Scale of MTCNN cell to minimum face size
        minl = min([height, width]) * m  # Image size to have a 12x12 region
        # correspond to minimum face size
        n_scales = math.ceil(math.log(12 / minl, factor))
        scales = [m * np.power(factor, n) for n in range(n_scales)]

        # Stage 1
        x1y1x2y2s = []
        for idx, scale in enumerate(scales):
            X = np.float32([
                image.resize(scale, interpolation=cv2.INTER_AREA).transpose(
                    1, 0, 2).scaled(-1, 1)
            ])
            y = [v.transpose(0, 2, 1, 3) for v in self.pnet.predict(X)]
            x1y1x2y2s_current = calculate_boxes(clf=y[0][0],
                                                reg=y[1][0],
                                                scale=scale,
                                                threshold=threshold,
                                                cell_size=12)
            if len(x1y1x2y2s_current) == 0:
                continue
            x1y1x2y2s.append(x1y1x2y2s_current[nms(x1y1x2y2s_current, 0.5,
                                                   "Union")])
        if len(x1y1x2y2s) == 0:
            return []
        x1y1x2y2s = np.concatenate(x1y1x2y2s)
        apply_regression(x1y1x2y2s)

        x1y1x2y2s = x1y1x2y2s[nms(x1y1x2y2s, 0.7, "Union")]

        # Stage 2
        pad_and_square(x1y1x2y2s, width=width, height=height, pad=0.1)
        x1y1x2y2s = run_stage(self.rnet,
                              image,
                              x1y1x2y2s,
                              threshold=threshold,
                              cell_size=24)
        if len(x1y1x2y2s) == 0:
            return []
        x1y1x2y2s = x1y1x2y2s[nms(x1y1x2y2s, 0.7, "Union")]

        # Stage 3
        pad_and_square(x1y1x2y2s, width=width, height=height, pad=0.1)
        x1y1x2y2s = run_stage(self.onet,
                              image,
                              x1y1x2y2s,
                              threshold=threshold,
                              cell_size=48)
        if len(x1y1x2y2s) == 0:
            return []
        x1y1x2y2s = x1y1x2y2s[nms(x1y1x2y2s, 0.5, "Min")]
        return [
            core.Annotation(
                selection=core.Selection(points=[[x1, y1], [x2, y2]]),
                category=self.annotation_config['face'],
                score=s
            ) for x1, y1, x2, y2, s in x1y1x2y2s[:, :5]
        ]
