#!/bin/bash
SED="sed -i.bak"
modules=(keras object_detection dataset backbone dataloader hparams_config utils efficientdet_arch nms_np det_model_fn coco_metric visualize inference iou_utils)
for module in ${modules[@]}
do
    $SED "s/^from $module/from .$module/g" mira/thirdparty/automl/efficientdet/*.py
    $SED "s/^import $module/from . import $module/g" mira/thirdparty/automl/efficientdet/*.py
    $SED "s/^import $module/from .. import $module/g" mira/thirdparty/automl/efficientdet/*/*.py
    $SED "s/^from $module import/from ..$module import/g" mira/thirdparty/automl/efficientdet/*/*.py
    if [ ! -f "mira/thirdparty/automl/efficientdet/$module.py" ]; then
        $SED "s/^from $module import/from . import/g" mira/thirdparty/automl/efficientdet/$module/*.py
    fi
done
$SED "s/\[bs, width, height, -1\]/[-1, width, height, self.config.num_classes * self.config.num_scales * len(self.config.aspect_ratios)]/g" mira/thirdparty/automl/efficientdet/keras/train_lib.py
$SED "s/\[bs, width, height, -1, self.config.num_classes\]/[-1, width, height, self.config.num_scales * len(self.config.aspect_ratios), self.config.num_classes]/g" mira/thirdparty/automl/efficientdet/keras/train_lib.py
$SED "s/import yaml//g" mira/thirdparty/automl/efficientdet/*.py
$SED "s/import yaml//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/import yaml//g" mira/thirdparty/automl/efficientnetv2/*.py
$SED "s/import tensorflow_hub as hub//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/import neural_structured_learning as nsl//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/import tensorflow_model_optimization as tfmot//g" mira/thirdparty/automl/efficientdet/*/*.py
$SED "s/tfmot.sparsity.keras.prune_low_magnitude/None/g" mira/thirdparty/automl/efficientdet/*/*.py
rm mira/thirdparty/**/**/*.bak
touch mira/thirdparty/automl/__init__.py
