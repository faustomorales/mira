import random
import typing
import logging

import numpy as np
import albumentations as A
import typing_extensions as tx

from . import experimental as mce
from . import utils as mcu

LOGGER = logging.getLogger(__name__)

BboxParams = A.BboxParams(format="pascal_voc", label_fields=["categories"])

AugmentedResult = tx.TypedDict(
    "AugmentedResult",
    {
        "image": np.ndarray,
        "bboxes": typing.List[typing.Tuple[int, int, int, int]],
        "categories": typing.List[str],
    },
)


# pylint: disable=too-few-public-methods
class AugmenterProtocol(tx.Protocol):
    """A protocol defining how we expect augmentation
    pipelines to behave. bboxes is expected to be in
    pascal_voc (or x1, y1, x2, y2) format."""

    def __call__(
        self,
        image: np.ndarray,
        bboxes: typing.List[typing.Tuple[int, int, int, int]],
        categories: typing.List[str],
    ) -> AugmentedResult:
        pass


class RandomCropBBoxSafe(A.DualTransform):
    """Crop an image and avoid splitting bounding boxes.

    Args:
        width: The maximum width of the cropped area.
        height: The maximum height of the cropped area.
        p (float): probability of applying the transform. Default: 1.
        prob_box: The probability of targeting a crop containing a box.
        cache: A dict-like object to use as a cache for computing
            safe crops.
    Targets:
        image, bboxes
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        width,
        height,
        always_apply=False,
        p=1.0,
        prob_box: float = 0.0,
        min_size=1,
        cache=None,
    ):
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.cache = cache
        self.min_size = min_size
        self.prob_box = prob_box

    def apply(self, img, **params):
        x_min, y_min, x_max, y_max = [
            params[k] for k in ["x_min", "y_min", "x_max", "y_max"]
        ]
        img_h, img_w = img.shape[:2]
        x_min_i, x_max_i = round(x_min * img_w), round(x_max * img_w)
        y_min_i, y_max_i = round(y_min * img_h), round(y_max * img_h)
        crop = img[y_min_i:y_max_i, x_min_i:x_max_i]
        return crop

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        bboxes = (
            (
                (
                    np.array([b[:4] for b in params["bboxes"]])
                    * np.array([img_w, img_h, img_w, img_h])
                )
                if len(params["bboxes"]) > 0
                else np.empty((0, 4))
            )
            .round()
            .astype("int32")
        )
        crops = mce.find_acceptable_crops(
            include=bboxes,
            width=img_w,
            height=img_h,
            max_width=self.width,
            max_height=self.height,
            cache=self.cache,
        )
        # Make sure we have crops that are at least 1px wide.
        crops = crops[(crops[:, 2:] - crops[:, :2]).min(axis=1) > self.min_size]
        if len(bboxes) > 0 and len(crops) > 0 and random.random() < self.prob_box:
            contains_box = mcu.compute_coverage(bboxes, crops).max(axis=0) > 0
            crops = crops[contains_box] if contains_box.any() else crops
        if len(crops) == 0:
            raise ValueError("Failed to find a suitable crop.")
        crop_idx = round(random.random() * (max(crops.shape[0] - 1, 0)))
        LOGGER.debug("Selecting %s from list of %s crops.", crop_idx, len(crops))
        crop = crops[crop_idx]
        crop = crop / (img_w, img_h, img_w, img_h)
        return dict(zip(["x_min", "y_min", "x_max", "y_max"], crop))

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def apply_to_bbox(self, bbox, **params):
        x_min, y_min, x_max, y_max = [
            params[k] for k in ["x_min", "y_min", "x_max", "y_max"]
        ]
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        scale_x = 1 / crop_w
        scale_y = 1 / crop_h
        x1, y1, x2, y2 = [
            min(max(value - offset, 0), max_dim) * scale
            for value, offset, max_dim, scale in zip(
                bbox,
                [x_min, y_min, x_min, y_min],
                [crop_w, crop_h, crop_w, crop_h],
                [scale_x, scale_y, scale_x, scale_y],
            )
        ]
        return (x1, y1, x2, y2)

    def get_transform_init_args_names(self):
        return ("width", "height")

    def apply_to_keypoint(self, *args, **kwargs):
        raise NotImplementedError("Keypoints are not supported.")
