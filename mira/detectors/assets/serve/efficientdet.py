# type: ignore

import torch
import torchvision
import effdet


class EfficientDet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = effdet.create_model(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            bench_task="",
            pretrained=False,
            pretrained_backbone=False,
            image_size=(INPUT_HEIGHT, INPUT_WIDTH),
        )
        self.config = {"anchors": effdet.anchors.Anchors.from_config(self.model.config)}
        self.resize = torchvision.transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH))

    def forward(self, x):
        resized = self.resize(x)
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
            num_levels=NUM_LEVELS,  # type: ignore
            num_classes=NUM_CLASSES,  # type: ignore
            max_detection_points=MAX_DETECTION_POINTS,  # type: ignore
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
            max_det_per_image=MAX_DET_PER_IMAGE,  # type: ignore
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
