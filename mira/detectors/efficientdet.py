# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
import os
import types
import logging
import tensorflow as tf
import numpy as np

try:
    from ..thirdparty.automl.efficientdet.keras import (
        postprocess,
        util_keras,
        anchors,
        train_lib,
        label_util,
    )
    from ..thirdparty.automl.efficientdet import (
        hparams_config,
        utils,
    )
except ImportError:
    # Do this so that the docs build since we don't
    # get a chance to run the packaging/package-automl.sh script.
    train_lib = types.SimpleNamespace(EfficientDetNetTrain=object)  # type: ignore
    label_util = types.SimpleNamespace(coco={1: "1", 2: "2"})  # type: ignore


from .. import core as mc
from .detector import Detector
from ..utils import get_file

log = logging.getLogger(__name__)

# pylint: disable=too-many-ancestors,bad-super-call,abstract-method
class EfficientDetNetTrain(train_lib.EfficientDetNetTrain):
    """A utility subclass purely to avoid having to set model_dir
    on the config object."""

    def __init__(self, *args, **kwargs):
        super(train_lib.EfficientDetNetTrain, self).__init__(*args, **kwargs)


def build_config(annotation_config, model_name="efficientdet-d0", input_shape=None):
    """Build a Google Automl config dict from an annotation
    configuration.

    Args:
        annotation_config: The annotation configuration.
        model_name: The model name to build.
        model_dir: The directory in which to save training snapshots.
    """
    # Parse and override hparams
    config = hparams_config.get_detection_config(model_name)
    config.num_classes = len(annotation_config) + 1
    # Parse image size in case it is in string format.
    config.img_summary_steps = None
    config.image_size = (
        input_shape[:2]
        if input_shape is not None
        else utils.parse_image_size(config.image_size)
    )
    config.input_shape = input_shape or (*config.image_size, 3)
    config.steps_per_execution = 1
    config.batch_size = 2
    config.steps_per_epoch = 120000 // 64
    return config


def ckpt_to_weights(model, ckpt_path_or_file, decay=0.9998):
    """Restore variables from a given checkpoint.

    Args:
        model: the keras model to be restored.
        ckpt_path_or_file: the path or file for checkpoint.

    Raises:
        KeyError: if access unexpected variables.
    """
    ckpt_path_or_file = tf.train.latest_checkpoint(ckpt_path_or_file)
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_vars = util_keras.get_ema_vars(model)
    reader = tf.train.load_checkpoint(ckpt_path_or_file)
    shape_map = reader.get_variable_to_shape_map()
    weights = []
    for v in model.weights:
        ref = v.ref()
        if ref not in ema_vars:
            key = v.name.split(":")[0]
        else:
            key = util_keras.average_name(ema, ema_vars[ref])
        assert key in shape_map, f"Missing {key}."
        assert (
            shape_map[key] == v.shape
        ), f"{key} has shape {v.shape} in model but {shape_map[key]} in weights."
        del shape_map[key]
        weights.append(reader.get_tensor(key))
    return weights


def build_model(config, **kwargs) -> tf.keras.Model:
    """Build a training model using a Google AutoML config dict."""
    input_shape = config.input_shape
    assert all(
        s is not None for s in input_shape
    ), "You must specify a fixed input shape (e.g., (512, 512, 3))."
    model = EfficientDetNetTrain(config=config, **kwargs)
    model.build((None, *input_shape))
    model(tf.keras.layers.Input(input_shape), True)
    return model


def convert_ckpt_to_h5(model_name, ckpt_dir, h5_path, notop_h5_path):
    """Convert the Google weights in CKPT format to
    vanilla Keras h5. Must be run on the first model built in the session.

    Args:
        model_name: The name of the model (e.g., "efficientdet-d0")
        ckpt_dir: The directory for the checkpoint.
        h5_path: Where to save the full model weights.
        notop_h5_path: Where to save the backbone weights.
    """
    config = build_config(GoogleCocoAnnotationConfiguration, model_name)
    model = build_model(config)

    # This has to run on the first model build to work.
    model.set_weights(ckpt_to_weights(model=model, ckpt_path_or_file=ckpt_dir))
    model.save_weights(h5_path)
    model.layers[0].save_weights(notop_h5_path)


def pluck_and_concatenate(vectors, key):
    """Utility function for plucking values from a list
    of dictionaries and concatenating the result."""
    return tf.concat(
        list(map(lambda t: tf.expand_dims(t[key], axis=0), vectors)), axis=0
    )


GoogleCocoAnnotationConfiguration = mc.AnnotationConfiguration(
    [
        label_util.coco.get(idx, f"placeholder_{idx}")
        for idx in range(1, max(label_util.coco))
    ]
)

WEIGHT_FILES = {
    "efficientdet-d0": {
        "top": {
            "hash": "0856fd2984caaff61c89b77d0d309cc756f71d9519644ba16daadc90c26ae491",
            "fname": "efficientdet-d0.h5",
        },
        "notop": {
            "hash": "83a64d8c0d2196aa4986bcc27fdc3f9ae235e31bd50975e88ea384cf0e50186c",
            "fname": "efficientdet-d0_notop.h5",
        },
    },
    "efficientdet-d1": {
        "top": {
            "hash": "8eb2128040b4e6637961391449d5d990cd6826ac5023fc971314e4d323048a5a",
            "fname": "efficientdet-d1.h5",
        },
        "notop": {
            "hash": "d55acfb94904fbc8de4e930a174d082caffc104dce4dcffb098a7ae1a73ed97c",
            "fname": "efficientdet-d1_notop.h5",
        },
    },
    "efficientdet-d2": {
        "top": {
            "hash": "86161d4844156e301c0039e2cde0a99511f2a392a71fdc1113b0947987358cd3",
            "fname": "efficientdet-d2.h5",
        },
        "notop": {
            "hash": "6674a4d374031c8f7fe5fc02c150f29561dcda40d8df362c40457e121ad67b9d",
            "fname": "efficientdet-d2_notop.h5",
        },
    },
    "efficientdet-d3": {
        "top": {
            "hash": "a42f5ee641600f8eeba1d3017b298db35dff95e60edfcdabd59588fa39ede370",
            "fname": "efficientdet-d3.h5",
        },
        "notop": {
            "hash": "0494719bdfc59b336aade8ddacd1c96045532725a0fa681287a0f288c91e34c2",
            "fname": "efficientdet-d3_notop.h5",
        },
    },
    "efficientdet-d4": {
        "top": {
            "hash": "4657d2fdee14d522be6615f06a523f37d48b6649c81b407b3ecd1a48393b14e1",
            "fname": "efficientdet-d4.h5",
        },
        "notop": {
            "hash": "48aaf3dd96e266845098b9ff643ff460208b89b9a5058c3b210668953fb82454",
            "fname": "efficientdet-d4_notop.h5",
        },
    },
    "efficientdet-d5": {
        "top": {
            "hash": "5790b57f4285c54640110907134d7a95599169d4626f42baa9285125fd5fa9b8",
            "fname": "efficientdet-d5.h5",
        },
        "notop": {
            "hash": "8d50a54773b842ebdb3dd7f13007510655b468813e0f3ee1df190ec3d10aa54c",
            "fname": "efficientdet-d5_notop.h5",
        },
    },
    "efficientdet-d6": {
        "top": {
            "hash": "9801979880eb07b188f1308ae704dcec3596e1205690954e404761f52a991da9",
            "fname": "efficientdet-d6.h5",
        },
        "notop": {
            "hash": "3d3eb5eb0d7baadb3869597c4a221bc559bc61150be2539931dae26f9f1d4d62",
            "fname": "efficientdet-d6_notop.h5",
        },
    },
    "efficientdet-d7": {
        "top": {
            "hash": "32621a72f97dbd8f286aae80f8af47a179996b651bc3b88ae7a494c3593366a9",
            "fname": "efficientdet-d7.h5",
        },
        "notop": {
            "hash": "86ff51bfb941c1a649ca6a9c67f59369dbc7cf883298dea062ec2841b6f07a2d",
            "fname": "efficientdet-d7_notop.h5",
        },
    },
}


class EfficientDet(Detector):
    """A detector wrapping
    `EfficientDet <https://arxiv.org/abs/1911.09070>`_
    using the
    `official repository <https://github.com/google/automl/tree/master/efficientdet>`_.

    The pretrained weights were converted from the checkpoints in the official repository
    using the following code. You have to restart the Python session each time in order to ensure the layer
    names generated by Keras match those in the checkpoint.


    .. code-block:: python

        import os
        import mira.detectors.efficientdet as mde

        model_name = "efficientdet-d7"
        mde.convert_ckpt_to_h5(
            model_name,
            ckpt_dir=os.path.expanduser(os.path.join("~", "downloads", model_name)),
            h5_path=os.path.expanduser(os.path.join("~", "downloads", f"{model_name}.h5")),
            notop_h5_path=os.path.expanduser(os.path.join("~", "downloads", f"{model_name}_notop.h5"))
        )

    Args:
        annotation_config: The annotation configuration to use for detection
        input_shape: Tuple of (height, width, n_channels)
        size: One of `efficient-d0` through `efficientdet-d7`.
        pretrained_backbone: Whether to use a pretrained backbone for the model
        pretrained_top: Whether to use the pretrained full model (only
            supported for where `annotation_config` is
            `mira.detectors.efficientdet.GoogleCOCOAnnotationConfiguration`)

    Attributes:
        model: The base Keras model containing the weights for feature
            extraction and bounding box regression / classification model.
            You should use this model for loading and saving weights.
        backbone: The backbone for the model (all layers except for final
            classification and regression).
    """

    def __init__(
        self,
        annotation_config=GoogleCocoAnnotationConfiguration,
        size: str = "efficientdet-d0",
        input_shape=None,
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        **kwargs,
    ):
        assert (
            not pretrained_top or annotation_config == GoogleCocoAnnotationConfiguration
        ), (
            "pretrained_top requires annotation configuration "
            "to be mira.detectors.efficientdet.GoogleCocoAnnotationConfiguration"
        )
        self.annotation_config = annotation_config
        self.config = build_config(
            annotation_config=annotation_config,
            model_name=size,
            input_shape=input_shape,
        )
        self.model = build_model(config=self.config, **kwargs)
        self.backbone = self.model.layers[0]
        # self.model = build_model(self.training_model, config=self.config)
        self.anchor_labeler = anchors.AnchorLabeler(
            anchors.Anchors(
                self.config.min_level,
                self.config.max_level,
                self.config.num_scales,
                self.config.aspect_ratios,
                self.config.anchor_scale,
                self.config.image_size,
            ),
            self.config.num_classes,
        )
        if pretrained_backbone and pretrained_top:
            log.warning(
                "pretrained_top makes pretrained_backbone "
                "redundant. Disabling pretrained_backbone."
            )
            pretrained_backbone = False
        if pretrained_backbone or pretrained_top:
            config = WEIGHT_FILES[size]["top" if pretrained_top else "notop"]
            weights_path = get_file(
                fname=config["fname"],
                origin=f"https://github.com/faustomorales/mira/releases/download/file-storage/{config['fname']}",
                file_hash=config["hash"],
                cache_subdir=os.path.join("weights", "efficientdet"),
                hash_algorithm="sha256",
            )
            if pretrained_backbone:
                self.backbone.load_weights(weights_path)
            if pretrained_top:
                self.model.load_weights(weights_path)
        self.compile()

    def compile(self):
        self.model.compile(
            steps_per_execution=self.config.steps_per_execution,
            optimizer=train_lib.get_optimizer(self.config.as_dict()),
            loss={
                train_lib.BoxLoss.__name__: train_lib.BoxLoss(
                    self.config.delta, reduction=tf.keras.losses.Reduction.NONE
                ),
                train_lib.BoxIouLoss.__name__: train_lib.BoxIouLoss(
                    self.config.iou_loss_type,
                    self.config.min_level,
                    self.config.max_level,
                    self.config.num_scales,
                    self.config.aspect_ratios,
                    self.config.anchor_scale,
                    self.config.image_size,
                    reduction=tf.keras.losses.Reduction.NONE,
                ),
                train_lib.FocalLoss.__name__: train_lib.FocalLoss(
                    self.config.alpha,
                    self.config.gamma,
                    label_smoothing=self.config.label_smoothing,
                    reduction=tf.keras.losses.Reduction.NONE,
                ),
                tf.keras.losses.SparseCategoricalCrossentropy.__name__: tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
                ),
            },
        )

    def invert_targets(
        self,
        y,
        input_shape,
        threshold=0.5,
        **kwargs,
    ):
        mode = kwargs.get("mode", "global")
        if isinstance(y, dict):
            cls_outputs, box_outputs = [
                [
                    y[f"{prefix}_targets_{level}"]
                    for level in range(self.config.min_level, self.config.max_level + 1)
                ]
                for prefix in ["cls", "box"]
            ]
            cls_outputs = [
                tf.cast(
                    tf.reshape(
                        tf.one_hot(
                            cls_output, self.config.num_classes, dtype=cls_output.dtype
                        ),
                        [
                            -1,
                            cls_output.shape[1],
                            cls_output.shape[2],
                            self.config.num_classes
                            * self.config.num_scales
                            * len(self.config.aspect_ratios),
                        ],
                    ),
                    tf.float32,
                )
                for cls_output in cls_outputs
            ]
        else:
            cls_outputs, box_outputs = y
        scales = np.ones(len(cls_outputs[0]))
        config_dict = self.config.as_dict()
        config_dict["nms_configs"]["score_thresh"] = threshold
        if mode == "global":
            f = postprocess.postprocess_global
        if mode == "per_class":
            f = postprocess.postprocess_per_class
        if mode == "combined":
            f = postprocess.postprocess_combined
        boxes, scores, classes, valid_len = f(
            config_dict, cls_outputs, box_outputs, scales
        )
        return [
            [
                mc.Annotation(
                    selection=mc.Selection(x1, y1, x2, y2),
                    # EfficientDet is 1-indexed so we subtract
                    # one to get back to a zero-indexed class.
                    category=self.annotation_config[int(c) - 1],
                    score=s,
                )
                for (y1, x1, y2, x2), c, s in zip(
                    current_boxes[:length],
                    current_classes[:length],
                    current_scores[:length],
                )
            ]
            for current_boxes, current_classes, current_scores, length in zip(
                boxes, classes, scores, valid_len
            )
        ]

    def compute_inputs(self, images):
        """Compute inputs from images."""
        assert (
            self.model.input_shape[-1] == 3
        ), "You must override compute_inputs for non-RGB images."
        return (np.float32(images) - self.config.mean_rgb) / self.config.stddev_rgb

    def compute_targets(self, annotation_groups, input_shape=None):
        """Compute target labels from a SceneCollection."""
        input_shape = input_shape or self.model.input_shape[1:]
        batch_size = len(annotation_groups)
        cls_targets = []
        box_targets = []
        num_positives = []
        for annotation_group in annotation_groups:
            # We re-order the columns because
            # `label_anchors` expects [y0, x0, y1, x1]
            bboxes = tf.convert_to_tensor(
                self.annotation_config.bboxes_from_group(annotation_group)[
                    :, [1, 0, 3, 2, 4]
                ],
                dtype=tf.float32,
            )
            boxes = bboxes[:, :-1]
            # EfficientDet uses the 0th class as
            # background so we adjust our classes to be
            # 1-indexed.
            classes = bboxes[:, -1:] + 1
            (
                cls_targets_current,
                box_targets_current,
                num_positives_current,
            ) = self.anchor_labeler.label_anchors(boxes, classes)
            cls_targets.append(cls_targets_current)
            box_targets.append(box_targets_current)
            num_positives.append(num_positives_current)
        labels = {}
        dtype = (
            tf.keras.mixed_precision.global_policy().compute_dtype
            if self.config.mixed_precision
            else None
        )
        for level in range(self.config.min_level, self.config.max_level + 1):
            labels[f"cls_targets_{level}"] = pluck_and_concatenate(cls_targets, level)
            labels[f"box_targets_{level}"] = pluck_and_concatenate(box_targets, level)
            if dtype is not None:
                labels[f"box_targets_{level}"] = tf.nest.map_structure(
                    lambda v: tf.cast(v, dtype=dtype), labels[f"box_targets_{level}"]
                )
        labels["mean_num_positives"] = tf.reshape(
            tf.tile(
                tf.reduce_mean(
                    tf.reshape(tf.concat(num_positives, axis=0), shape=[-1]),
                    keepdims=True,
                ),
                [
                    batch_size,
                ],
            ),
            [batch_size, 1],
        )
        return labels
