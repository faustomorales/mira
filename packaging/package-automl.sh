modules=(keras object_detection dataset backbone dataloader hparams_config utils efficientdet_arch nms_np det_model_fn coco_metric visualize inference iou_utils)
for module in ${modules[@]}
do
    sed -i '' "s/^from $module/from .$module/g" mira/thirdparty/automl/efficientdet/*.py
    sed -i '' "s/^import $module/from . import $module/g" mira/thirdparty/automl/efficientdet/*.py
    sed -i '' "s/^import $module/from .. import $module/g" mira/thirdparty/automl/efficientdet/*/*.py
    sed -i '' "s/^from $module import/from ..$module import/g" mira/thirdparty/automl/efficientdet/*/*.py
    if [ ! -f "mira/thirdparty/automl/efficientdet/$module.py" ]; then
        sed -i '' "s/^from $module import/from . import/g" mira/thirdparty/automl/efficientdet/$module/*.py
    fi
done
sed -i '' "s/\[bs, width, height, -1\]/[-1, width, height, self.config.num_classes * self.config.num_scales * len(self.config.aspect_ratios)]/g" mira/thirdparty/automl/efficientdet/keras/train_lib.py
sed -i '' "s/\[bs, width, height, -1, self.config.num_classes\]/[-1, width, height, self.config.num_scales * len(self.config.aspect_ratios), self.config.num_classes]/g" mira/thirdparty/automl/efficientdet/keras/train_lib.py
touch mira/thirdparty/automl/__init__.py
python -c "import mira.thirdparty.automl.efficientdet.keras.efficientdet_keras"