# type: ignore

import torch
import torchvision
import numpy as np
import mira.thirdparty.effdet as effdet
import omegaconf
import mira.detectors.common as mdc


class EfficientDet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = omegaconf.OmegaConf.merge(
            effdet.get_efficientdet_config(model_name=MODEL_NAME),
            omegaconf.OmegaConf.create(
                {
                    "min_level": MIN_LEVEL,
                    "max_level": MAX_LEVEL,
                    "num_levels": NUM_LEVELS,
                    "max_detection_points": MAX_DETECTION_POINTS,
                }
            ),
        )
        self.model = effdet.create_model_from_config(
            config=config,
            num_classes=NUM_CLASSES,
            bench_task="",
            pretrained=False,
            pretrained_backbone=False,
            image_size=(INPUT_HEIGHT, INPUT_WIDTH),
            max_det_per_image=MAX_DET_PER_IMAGE,
        )
        self.config = {"anchors": effdet.anchors.Anchors.from_config(self.model.config)}
        self.mean = torch.tensor(
            np.array([[[[0.485]], [[0.456]], [[0.406]]]]), dtype=torch.float32
        )
        self.std = torch.tensor(
            np.array([[[[0.229]], [[0.224]], [[0.225]]]]), dtype=torch.float32
        )

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x):
        resized, scales, sizes = mdc.resize(
            x,
            resize_method=RESIZE_METHOD,
            height=INPUT_HEIGHT,
            width=INPUT_WIDTH,
            base=128,
        )
        # Go from [sy, sx] to [sx, sy, sx, sy]
        scales = scales[:, [1, 0]].repeat((1, 2))
        sizes = sizes[:, [1, 0]].repeat((1, 2))
        resized = self.normalize(resized)
        if (
            self.model.config.image_size[0] != resized.shape[2]
            or self.model.config.image_size[1] != resized.shape[3]
        ):
            self.model.config = omegaconf.OmegaConf.merge(  # type: ignore
                self.model.config,
                omegaconf.OmegaConf.create(
                    {"image_size": (resized.shape[2], resized.shape[3])}
                ),
            )
            self.config["anchors"] = effdet.anchors.Anchors.from_config(
                self.model.config
            )
        y = self.model(resized)
        class_out, box_out = y
        class_out, box_out, indices, classes = effdet.bench._post_process(
            class_out,
            box_out,
            num_levels=self.model.config.num_levels,
            num_classes=self.model.config.num_classes,
            max_detection_points=self.model.config.max_detection_points,
        )
        detections = effdet.bench._batch_detection(
            class_out.shape[0],
            class_out.cpu(),
            box_out.cpu(),
            self.config["anchors"].boxes.cpu(),
            indices.cpu(),
            classes.cpu(),
            img_scale=None,
            img_size=None,
            max_det_per_image=self.model.config.max_det_per_image,
            soft_nms=True,
        )
        clipped = [
            group[:, :4].min(group_size.unsqueeze(0))
            for group, group_size in zip(detections, sizes)
        ]
        has_area = [((c[:, 3] - c[:, 1]) * (c[:, 2] - c[:, 0])) > 0 for c in clipped]
        return [
            {
                "boxes": boxes[box_has_area] * (1 / scaler),
                "labels": group[box_has_area, 5].type(torch.IntTensor),
                "scores": group[box_has_area, 4],
            }
            for group, boxes, box_has_area, scaler in zip(
                detections, clipped, has_area, scales
            )
        ]
