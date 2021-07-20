import random
import typing

import numpy as np
import albumentations as A
import typing_extensions as tx

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


class RandomCropToPaddedBBox(A.DualTransform):
    """Crop an image to have a bounding box centered within it. Useful for
    cases where labels are trusted in the area directly near a bounding box
    but not far from them.
    Args:
        width: The width of the cropped area.
        height: The height of the cropped area.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, bboxes
    Image types:
        uint8, float32
    """

    def __init__(self, width, height, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.width = width
        self.height = height

    def apply(self, img, **params):
        x_min, y_min = [params[k] for k in ["x_min", "y_min"]]
        img_h, img_w = img.shape[:2]
        x_min_i = max(int(min(x_min * img_w, img_w - self.width)), 0)
        y_min_i = max(int(min(y_min * img_h, img_h - self.height)), 0)
        x_max_i = x_min_i + self.width
        y_max_i = y_min_i + self.height
        crop = img[y_min_i:y_max_i, x_min_i:x_max_i]
        if x_max_i <= img_w and y_max_i <= img_h:
            return crop
        return np.pad(
            crop, ((0, max(y_max_i - img_h, 0)), (0, max(x_max_i - img_w, 0)), (0, 0))
        )

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        box_h, box_w = min(1, self.height / img_h), min(1, self.width / img_w)
        if (
            len(params["bboxes"]) == 0
        ):  # less likely, this class is for use with bboxes.
            # Choose a random region
            x_min = int(random.random() * (1 - box_w))
            y_min = int(random.random() * (1 - box_h))
            return {
                "x_min": x_min,
                "y_min": y_min,
            }
        x1b, y1b, x2b, y2b = params["bboxes"][
            round(random.random() * (len(params["bboxes"]) - 1))
        ][:4]
        xcb, ycb = (x1b + x2b) / 2, (y1b + y2b) / 2
        x_min = min(max(0, xcb - box_w / 2), 1 - box_w)
        y_min = min(max(0, ycb - box_h / 2), 1 - box_h)
        return {
            "x_min": x_min,
            "y_min": y_min,
        }

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def apply_to_bbox(self, bbox, **params):
        x_min, y_min, cols, rows = [
            params[k] for k in ["x_min", "y_min", "cols", "rows"]
        ]
        crop_w = self.width / cols
        crop_h = self.height / rows
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
