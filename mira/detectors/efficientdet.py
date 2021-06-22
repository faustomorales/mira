# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
import tensorflow as tf
import numpy as np

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

from .. import core as mc
from .detector import Detector


# pylint: disable=too-many-ancestors,bad-super-call,abstract-method
class EfficientDetNetTrain(train_lib.EfficientDetNetTrain):
    """A utility subclass purely to avoid having to set model_dir
    on the config object."""

    def __init__(self, *args, **kwargs):
        super(train_lib.EfficientDetNetTrain, self).__init__(*args, **kwargs)


def build_config(annotation_config, model_name="efficientdet-d0"):
    """Build a Google Automl config dict from an annotation
    configuration.

    Args:
        annotation_config: The annotation configuration.
        model_name: The model name to build.
        model_dir: The directory in which to save training snapshots.
    """
    # Parse and override hparams
    config = hparams_config.get_detection_config(model_name)
    config.num_classes = len(annotation_config)
    # Parse image size in case it is in string format.
    config.img_summary_steps = None
    config.image_size = utils.parse_image_size(config.image_size)
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
        if key not in shape_map:
            continue
        assert (
            shape_map[key] == v.shape
        ), f"{key} has shape {v.shape} in model but {shape_map[key]} in weights."
        del shape_map[key]
        weights.append(reader.get_tensor(key))
    return weights


def build_model(config, input_shape=None, **kwargs) -> tf.keras.Model:
    """Build a training model using a Google AutoML config dict."""
    input_shape = input_shape or (*config.image_size, 3)
    assert all(
        s is not None for s in input_shape
    ), "You must specify a fixed input shape (e.g., (512, 512, 3))."
    model = EfficientDetNetTrain(config=config, **kwargs)
    model.build((None, *input_shape))
    model(tf.keras.layers.Input(input_shape), True)
    return model


def convert_ckpt_to_h5(model_name):
    """Convert the CKPT format to vanilla Keras h5. Must be run
    on the first model built in the session. That limitation is why we
    do the conversion in the first place.

    Args:
        model_name: The name of the model (e.g., "efficientdet-d0")
    """
    config = hparams_config.get_detection_config(model_name)
    model = build_model(config)

    # This has to run on the first model build to work.
    model.set_weights(ckpt_to_weights(model=model, ckpt_path_or_file=model_name))
    model.save_weights(f"{model_name}.h5")


def pluck_and_concatenate(vectors, key):
    """Utility function for plucking values from a list
    of dictionaries and concatenating the result."""
    return tf.concat(
        list(map(lambda t: tf.expand_dims(t[key], axis=0), vectors)), axis=0
    )


GoogleCocoAnnotationConfiguration = mc.AnnotationConfiguration(
    [
        label_util.coco.get(idx, f"placeholder_{idx}")
        for idx in range(max(label_util.coco))
    ]
)


class EfficientDet(Detector):
    """A detector wrapping
    `EfficientDet <https://arxiv.org/abs/1911.09070>`_
    using the
    `official repository <https://github.com/google/automl/tree/master/efficientdet>`_.

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
        annotation_config,
        size: str = "efficientdet-d0",
        input_shape=None,
        **kwargs,
    ):
        self.annotation_config = annotation_config
        self.config = build_config(annotation_config=annotation_config, model_name=size)
        self.model = build_model(config=self.config, input_shape=input_shape, **kwargs)
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
                    selection=mc.Selection([[x1, y1], [x2, y2]]),
                    category=self.annotation_config[int(c)],
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
        # assert (
        #     self.model.input_shape[-1] == 3
        # ), "You must override compute inputs for non-RGB images."
        return (np.float32(images) - self.config.mean_rgb) / self.config.stddev_rgb

    def compute_targets(self, collection, input_shape):
        """Compute target labels from a SceneCollection."""
        batch_size = len(collection)
        cls_targets = []
        box_targets = []
        num_positives = []
        for scene in collection:
            # We re-order the columns because
            # `label_anchors` expects [y0, x0, y1, x1]
            bboxes = tf.convert_to_tensor(
                scene.bboxes()[:, [1, 0, 3, 2, 4]], dtype=tf.float32
            )
            boxes = bboxes[:, :-1]
            classes = bboxes[:, -1:]
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
