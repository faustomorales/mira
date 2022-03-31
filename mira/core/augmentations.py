import random
import typing
import logging

import numpy as np
import albumentations as A
import typing_extensions as tx

from . import experimental as mce
from . import utils as mcu

LOGGER = logging.getLogger(__name__)

BboxParams = A.BboxParams(
    format="pascal_voc", label_fields=["bbox_indices"], check_each_transform=False
)
KeypointParams = A.KeypointParams(
    format="xy", label_fields=["keypoint_indices"], remove_invisible=False
)

AugmentedResult = tx.TypedDict(
    "AugmentedResult",
    {
        "image": np.ndarray,
        "bboxes": typing.List[typing.Tuple[int, int, int, int]],
        "keypoints": typing.List[typing.Tuple[int, int]],
        "bbox_indices": typing.List[int],
        "keypoint_indices": typing.List[typing.Tuple[int, int]],
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
        keypoints: typing.List[typing.Tuple[int, int]],
        bbox_indices: typing.List[int],
        keypoint_indices: typing.List[typing.Tuple[int, int]],
    ) -> AugmentedResult:
        pass


def wiggle_crop(crop, include, exclude, img_w, img_h, width, height):
    """Wiggle a bbox-safe crop without overlapping new bboxes."""
    dx1, dy1, success1 = mce.search(
        x=crop[0],
        y=crop[1],
        exclude=exclude,
        include=include,
        max_height=img_h - crop[1],
        max_width=img_w - crop[0],
    )
    assert success1, "Unexpected failure when performing first-pass crop wiggle."
    offset1 = np.array(
        [
            # Don't go into the box.
            include[:, :2].min(axis=0) - crop[:2]
            if len(include) > 0
            else [np.inf, np.inf],
            # Don't let the bottom right edge go
            # further than the safe area.
            (np.array([dx1, dy1]) - [width, height]).clip(0),
        ]
    ).min(axis=0)
    offset1 = (np.random.uniform(size=2) * offset1).round().astype("int32")
    crop = crop + np.tile(offset1, 2)
    params2 = dict(
        x=img_w - crop[2],
        y=img_h - crop[3],
        exclude=(img_w, img_h, img_w, img_h) - exclude[:, [2, 3, 0, 1]],
        include=(img_w, img_h, img_w, img_h) - include[:, [2, 3, 0, 1]],
        max_width=crop[2],
        max_height=crop[3],
    )
    dx2, dy2, success2 = mce.search(**params2)
    if not success2:
        LOGGER.warning("Second pass crop wiggle failed with parameters: %s", params2)
        return crop
    offset2 = np.array(
        [
            crop[2:] - include[:, 2:].max(axis=0)
            if len(include) > 0
            else [np.inf, np.inf],
            (np.array([dx2, dy2]) - [width, height]).clip(0),
        ]
    ).min(axis=0)
    offset2 = (np.random.uniform(size=2) * offset2).round().astype("int32")
    crop = crop - np.tile(offset2, 2)
    return crop


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
        image, bboxes, keypoints
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
        wiggle=False,
        cache=None,
    ):
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.cache = cache
        self.min_size = min_size
        self.prob_box = prob_box
        self.wiggle = wiggle

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
        if len(bboxes) > 0 and len(crops) > 0:
            coverage = mcu.compute_coverage(bboxes, crops)
            if random.random() < self.prob_box:
                contains_box = coverage.max(axis=0) > 0
                coverage = coverage[:, contains_box] if contains_box.any() else coverage
                crops = crops[contains_box] if contains_box.any() else crops
        else:
            coverage = np.empty((len(bboxes), len(crops)))
        if len(crops) == 0:
            raise ValueError("Failed to find a suitable crop.")
        crop_idx = round(random.random() * (max(crops.shape[0] - 1, 0)))
        LOGGER.debug("Selecting %s from list of %s crops.", crop_idx, len(crops))
        crop = crops[crop_idx]
        if self.wiggle:
            crop = wiggle_crop(
                crop=crop,
                include=bboxes[coverage[:, crop_idx] == 1],
                exclude=bboxes[coverage[:, crop_idx] == 0],
                img_w=img_w,
                img_h=img_h,
                width=self.width,
                height=self.height,
            )
        crop = crop / (img_w, img_h, img_w, img_h)
        return dict(zip(["x_min", "y_min", "x_max", "y_max"], crop))

    @property
    def targets_as_params(self):
        return ["image", "bboxes", "keypoints"]

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
        return ("width", "height", "wiggle", "prob_box", "min_size", "cache")

    def apply_to_keypoint(self, keypoint, **params):
        dx = params["x_min"] * params["cols"]
        dy = params["y_min"] * params["rows"]
        return (keypoint[0] - dx, keypoint[1] - dy, *keypoint[2:])
