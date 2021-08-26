# type: ignore

import torch
import torchvision
import numpy as np
import effdet
import omegaconf


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
        self.resize = torchvision.transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH))
        self.mean = torch.tensor(
            np.array([[[[0.485]], [[0.456]], [[0.406]]]]), dtype=torch.float32
        )
        self.std = torch.tensor(
            np.array([[[[0.229]], [[0.224]], [[0.225]]]]), dtype=torch.float32
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        resized = self.normalize(self.resize(x))
        scales = torch.tensor(
            [
                [i.shape[2] / o.shape[2], i.shape[1] / o.shape[1]]
                for i, o in zip(x, resized)
            ]
        ).tile((1, 2))
        y = self.model(resized)
        class_out, box_out = y
        class_out, box_out, indices, classes = effdet.bench._post_process(
            class_out,
            box_out,
            num_levels=self.model.config.num_levels,
            num_classes=self.model.config.num_classes,
            max_detection_points=self.model.config.max_detection_points,
        )
        img_scale, img_size = None, None
        detections = effdet.bench._batch_detection(
            class_out.shape[0],
            class_out.cpu(),
            box_out.cpu(),
            self.config["anchors"].boxes.cpu(),
            indices.cpu(),
            classes.cpu(),
            img_scale,
            img_size,
            max_det_per_image=self.model.config.max_det_per_image,
            soft_nms=True,
        )
        return [
            {
                "boxes": group[:, :4] * scaler,
                "labels": group[:, 5].type(torch.IntTensor),
                "scores": group[:, 4],
            }
            for group, scaler in zip(detections, scales)
        ]
