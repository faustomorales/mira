# type: ignore

import torch
import torchvision
import mira.detectors.common as mdc
import mira.detectors.retinanet as mdr


class RetinaNetObjectDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_kwargs = BACKBONE_KWARGS
        self.anchor_kwargs = ANCHOR_KWARGS
        self.detector_kwargs = DETECTOR_KWARGS
        self.model = torchvision.models.detection.retinanet.RetinaNet(
            backbone=mdr.BACKBONE_TO_PARAMS[BACKBONE_NAME]["backbone_func"](
                **{
                    k: (
                        v
                        if k != "extra_blocks"
                        or isinstance(
                            v, torchvision.ops.feature_pyramid_network.ExtraFPNBlock
                        )
                        else mdr.EXTRA_BLOCKS_MAP[v]()  # type: ignore
                    )
                    for k, v in self.backbone_kwargs.items()
                }
            ),
            num_classes=NUM_CLASSES,
            anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **self.anchor_kwargs
            ),
            **self.detector_kwargs,
        )
        self.model.transform = mdc.convert_rcnn_transform(self.model.transform)

    def forward(self, x):
        return mdc.torchvision_serve_inference(self, x=x, resize_config=RESIZE_CONFIG)
