# type: ignore

import torch
import torchvision
import mira.detectors.common as mdc
import mira.detectors.fasterrcnn as mdf


class FasterRCNNObjectDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = mdf.ModifiedFasterRCNN(
            mdf.BACKBONE_TO_PARAMS[BACKBONE_NAME]["fpn_func"](
                **mdc.interpret_fpn_kwargs(
                    FPN_KWARGS,
                    extra_blocks_kwargs=mdf.BACKBONE_TO_PARAMS[BACKBONE_NAME].get(
                        "fpn_extra_blocks_kwargs"
                    ),
                )
            ),
            NUM_CLASSES,
            rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
                **ANCHOR_KWARGS
            ),
            **DETECTOR_KWARGS,
        )
        self.model.transform = mdc.convert_rcnn_transform(self.model.transform)

    def forward(self, x):
        return mdc.torchvision_serve_inference(self, x=x, resize_config=RESIZE_CONFIG)
