# type: ignore

import cv2
import torch
import numpy as np
import mira.core
import mira.detectors.segmentation as mds


class SMPObjectDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = mds.SMPWrapper(
            model=getattr(mds.smp, ARCH)(
                **BACKBONE_KWARGS,
                classes=NUM_CLASSES,
                encoder_weights=None,
            ),
            **DETECTOR_KWARGS,
        )
        self.preprocessing_fn = mds.smp.encoders.get_preprocessing_fn(
            **PREPROCESSING_KWARGS
        )

    def forward(self, x):
        x, scales, sizes = mira.core.resizing.resize(x, resize_config=RESIZE_CONFIG)
        x = (
            self.preprocessing_fn(x.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .to(torch.float32)
        )
        y = self.model(x)
        records = [
            mira.core.utils.flatten(
                [
                    [
                        dict(
                            polygon=(contour[:, 0] / scale).tolist(),
                            label=classIdx + 1,
                            score=catmap[
                                contour[:, 0, 1]
                                .min(axis=0) : contour[:, 0, 1]
                                .max(axis=0)
                                + 1,
                                contour[:, 0, 0].min() : contour[:, 0, 0].max() + 1,
                            ].max()
                            if np.product(
                                contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0)
                            )
                            > 0
                            else BASE_THRESHOLD,
                        )
                        for contour in sorted(
                            cv2.findContours(
                                (catmap > BASE_THRESHOLD).astype("uint8"),
                                mode=cv2.RETR_LIST,
                                method=cv2.CHAIN_APPROX_SIMPLE,
                            )[0],
                            key=cv2.contourArea,
                            reverse=True,
                        )[:MAX_DETECTIONS]
                        if (contour[:, 0].min(axis=0) < limits).all()
                    ]
                    for classIdx, catmap in enumerate(
                        segmap["map"].detach().cpu().numpy()
                    )
                ]
            )
            for segmap, limits, scale in zip(
                y["output"],
                sizes.cpu().numpy(),
                scales.cpu().numpy(),
            )
        ]
        return [
            {
                ko: [r[ki] for r in group]
                for ki, ko in [
                    ("polygon", "polygons"),
                    ("score", "scores"),
                    ("label", "labels"),
                ]
            }
            for group in records
        ]
