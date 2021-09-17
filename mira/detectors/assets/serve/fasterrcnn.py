# type: ignore

import torch
import torchvision


class LastLevelNoop(torchvision.ops.feature_pyramid_network.ExtraFPNBlock):
    """
    A noop extra FPN block.
    """

    def forward(self, x, y, names):
        return x, names


BACKBONE_TO_CONSTRUCTOR = {
    "resnet50": torchvision.models.detection.backbone_utils.resnet_fpn_backbone,
    "mobilenet_large": torchvision.models.detection.backbone_utils.mobilenet_backbone,
    "mobilenet_large_320": torchvision.models.detection.backbone_utils.mobilenet_backbone,
}

EXTRA_BLOCKS_MAP = {
    "lastlevelmaxpool": torchvision.ops.feature_pyramid_network.LastLevelMaxPool,
    "noop": LastLevelNoop,
}


class FasterRCNNObjectDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.detection.FasterRCNN(
            backbone=BACKBONE_TO_CONSTRUCTOR[BACKBONE_NAME](
                **{
                    k: (
                        v
                        if k != "extra_blocks"
                        or isinstance(
                            v, torchvision.ops.feature_pyramid_network.ExtraFPNBlock
                        )
                        else EXTRA_BLOCKS_MAP[v]()
                    )
                    for k, v in BACKBONE_KWARGS.items()
                }
            ),
            num_classes=NUM_CLASSES,
            **DETECTOR_KWARGS,
            rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **ANCHOR_KWARGS
            )
        )
        self.model.transform.fixed_size = (INPUT_HEIGHT, INPUT_WIDTH)  # type: ignore
        self.model.transform.min_size = (min(INPUT_WIDTH, INPUT_HEIGHT),)  # type: ignore
        self.model.transform.max_size = max(INPUT_HEIGHT, INPUT_WIDTH)  # type: ignore

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
