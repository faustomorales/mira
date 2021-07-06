# type: ignore

import torch
import torchvision


BACKBONE_TO_CONSTRUCTOR = {
    "resnet50": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "mobilenet_large": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "mobilenet_large_320": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
}


class FasterRCNNObjectDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = BACKBONE_TO_CONSTRUCTOR[BACKBONE_NAME](
            pretrained=False, num_classes=NUM_CLASSES, pretrained_backbone=False
        )
        self.model.transform.fixed_size = (INPUT_HEIGHT, INPUT_WIDTH)  # type: ignore
        self.model.transform.min_size = (min(INPUT_WIDTH, INPUT_HEIGHT),)  # type: ignore
        self.model.transform.max_size = max(INPUT_HEIGHT, INPUT_WIDTH)  # type: ignore

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
