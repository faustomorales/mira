import torch
import effdet
import omegaconf
import numpy as np


from .. import datasets as mds
from .. import core as mc
from .detector import Detector


class EfficientDet(Detector):
    """A wrapper for EfficientDet as implemented in the effdet package."""

    def __init__(
        self,
        annotation_config=mds.COCOAnnotationConfiguration90,
        model_name: str = "tf_efficientdet_d0",
        pretrained_backbone: bool = True,
        pretrained_top: bool = False,
        device="cpu",
    ):
        super().__init__(device=device)
        default_config = effdet.get_efficientdet_config(model_name=model_name)
        self.annotation_config = annotation_config
        self.model = effdet.create_model(
            model_name=model_name,
            num_classes=len(annotation_config),
            bench_task="",
            pretrained=pretrained_top,
            pretrained_backbone=pretrained_backbone,
            image_size=(None, None),
        ).to(self.device)
        self.set_input_shape(
            width=default_config.image_size[1], height=default_config.image_size[0]
        )

    def set_input_shape(self, width, height):
        self.model.config = omegaconf.OmegaConf.merge(  # type: ignore
            self.model.config,
            omegaconf.OmegaConf.create({"image_size": (height, width)}),
        )
        self.anchors = effdet.anchors.Anchors.from_config(self.model.config)
        self.training_model = effdet.DetBenchTrain(self.model).to(self.device)

    @property
    def input_shape(self):
        return tuple(self.model.config.image_size) + (3,)  # type: ignore

    def compute_inputs(self, images):
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        images = (np.float32(images) - mean) / std
        return (
            torch.tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

    def compute_targets(self, annotation_groups):
        bboxes = [
            self.annotation_config.bboxes_from_group(g)[
                : self.model.config.max_det_per_image  # type: ignore
            ]
            for g in annotation_groups
        ]
        bboxes = [b + [[0, 0, 0, 0, 1]] for b in bboxes]
        bboxes = [
            np.pad(
                b,
                ((0, self.model.config.max_det_per_image - len(b)), (0, 0)),  # type: ignore
                mode="constant",
                constant_values=-1,
            )[:, [1, 0, 3, 2, 4]]
            for b in bboxes
        ]  # (ymin, xmin, ymax, xmax, class)
        return {
            "bbox": torch.tensor([b[:, :4] for b in bboxes], dtype=torch.float32).to(
                self.device
            ),
            "cls": torch.tensor([b[:, -1] for b in bboxes]).to(self.device),
        }

    # pylint: disable=protected-access
    def invert_targets(self, y, threshold=0.5, **kwargs):
        config = self.model.config
        class_out, box_out = y
        class_out, box_out, indices, classes = effdet.bench._post_process(
            class_out,
            box_out,
            num_levels=config.num_levels,  # type: ignore
            num_classes=config.num_classes,  # type: ignore
            max_detection_points=config.max_detection_points,  # type: ignore
        )
        img_scale, img_size = None, None
        detections = effdet.bench._batch_detection(
            class_out.shape[0],
            class_out,
            box_out,
            self.anchors.boxes.to(self.device),
            indices,
            classes,
            img_scale,
            img_size,
            max_det_per_image=config.max_det_per_image,  # type: ignore
            soft_nms=True,
        )
        return [
            [
                mc.Annotation(
                    category=self.annotation_config[int(labelIdx) - 1],
                    selection=mc.Selection(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=score,
                )
                for x1, y1, x2, y2, score, labelIdx in group.detach().cpu().numpy()
                if score > threshold
            ]
            for group in detections
        ]
