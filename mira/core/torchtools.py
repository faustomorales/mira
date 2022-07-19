import types
import typing
import random
import logging
import tqdm

try:
    import timm
    import timm.optim
    import timm.scheduler
except ImportError:
    timm = None  # type: ignore
try:
    import torch
except ImportError:
    torch = None  # type: ignore
import cv2
import numpy as np
import typing_extensions as tx

try:
    import torchvision.transforms.functional as tvtf
except ImportError:
    tvtf = None  # type: ignore

from . import annotation, scene

LOGGER = logging.getLogger()
DEFAULT_SCHEDULER_PARAMS = dict(
    sched="cosine",
    min_lr=1e-3,
    warmup_lr=0,
    warmup_epochs=0,
    cooldown_epochs=0,
    epochs=10,
    lr_cycle_decay=1,
    lr_cycle_limit=1e5,
    lr_cycle_mul=1,
)

DEFAULT_OPTIMIZER_PARAMS = dict(lr=1e-2, opt="sgd", weight_decay=4e-5)

InputType = typing.TypeVar("InputType")
FixedSizeConfig = tx.TypedDict(
    "FixedSizeConfig",
    {"method": tx.Literal["fit", "pad", "force"], "width": int, "height": int},
)
VariableSizeConfig = tx.TypedDict(
    "VariableSizeConfig", {"method": tx.Literal["pad_to_multiple"], "base": int}
)
ResizeConfig = typing.Union[
    FixedSizeConfig,
    VariableSizeConfig,
]
TrainItem = typing.NamedTuple(
    "TrainItem",
    [
        ("split", tx.Literal["train", "val"]),
        ("index", int),
        ("transform", np.ndarray),
        ("scene", scene.Scene),
    ],
)


def train(
    model: "torch.nn.Module",
    loss: typing.Callable[[typing.List[InputType]], "torch.Tensor"],
    training: typing.List[InputType],
    validation: typing.List[InputType] = None,
    batch_size: int = 1,
    augment: typing.Callable[[typing.List[InputType]], typing.List[InputType]] = None,
    epochs=100,
    on_epoch_start: typing.Callable = None,
    on_epoch_end: typing.Callable[[typing.List[dict]], dict] = None,
    shuffle=True,
    optimizer_params=None,
    scheduler_params=None,
    clip_grad_norm_params=None,
):
    """Run training job.
    Args:
        model: The model that we're training.
        loss: A function to compute the loss for a batch.
        training: The collection of training images
        validation: The collection of validation images
        batch_size: The batch size to use for training
        augmenter: The augmenter for generating samples
        epochs: The number of epochs to train.
        on_epoch_start: A callback to run when starting a new epoch.
        on_epoch_end: A callback to run when finishing an epoch.
        shuffle: Whether to shuffle the training data on each epoch.
        optimizer_params: Passed to timm.optim.create_optimizer_v2 to build
            the optimizer.
        scheduler_params: Passed to timm.scheduler.create_scheduler to build
            the scheduler.
    """
    assert timm is not None, "timm is required for this function"
    assert torch is not None, "torch is required for this function."
    optimizer = timm.optim.create_optimizer_v2(
        model, **(optimizer_params or DEFAULT_OPTIMIZER_PARAMS)
    )
    scheduler, _ = timm.scheduler.create_scheduler(
        types.SimpleNamespace(**(scheduler_params or DEFAULT_SCHEDULER_PARAMS)),
        optimizer=optimizer,
    )
    train_index = np.arange(len(training)).tolist()
    summaries: typing.List[typing.Dict[str, typing.Any]] = []
    terminated = False
    try:
        for epoch in range(epochs):
            with tqdm.trange(len(training) // batch_size) as t:
                model.train()
                t.set_description(f"Epoch {epoch + 1} / {epochs}")
                scheduler.step(
                    epoch=epoch,
                    metric=None
                    if not summaries
                    else summaries[-1].get("val_loss", summaries[-1]["loss"]),
                )
                if on_epoch_start:
                    on_epoch_start()
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    end = min(start + batch_size, len(train_index))
                    batch = [training[train_index[idx]] for idx in range(start, end)]
                    if augment:
                        batch = augment(batch)
                    optimizer.zero_grad()
                    batch_loss = loss(batch)
                    batch_loss.backward()
                    if clip_grad_norm_params is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), **clip_grad_norm_params
                        )
                    cum_loss += batch_loss.detach().cpu().numpy()
                    avg_loss = cum_loss / end
                    optimizer.step()
                    t.set_postfix(loss=avg_loss)
                    t.update()
                summaries.append({"loss": avg_loss})
                if validation:
                    summaries[-1]["val_loss"] = np.sum(
                        [
                            loss(
                                [
                                    validation[idx]
                                    for idx in range(
                                        vstart,
                                        min(vstart + batch_size, len(validation)),
                                    )
                                ]
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            for vstart in range(0, len(validation), batch_size)
                        ]
                    ) / len(validation)
                summaries[-1]["lr"] = next(g["lr"] for g in optimizer.param_groups)
                if on_epoch_end:
                    try:
                        summaries[-1] = {**summaries[-1], **on_epoch_end(summaries)}
                    except StopIteration:
                        terminated = True
                t.set_postfix(**summaries[-1])
            if terminated:
                break
    except KeyboardInterrupt:
        LOGGER.warning("Terminating early due to keyboard interrupt.")
        return summaries
    return summaries


ArrayType = typing.TypeVar("ArrayType", torch.Tensor, np.ndarray)


def fit(
    image: ArrayType, height: int, width: int, force: bool
) -> typing.Tuple[ArrayType, typing.Tuple[float, float], typing.Tuple[int, int]]:
    """Fit an image to a specific size, padding where necessary to maintain
    aspect ratio.

    Args:
        image: A tensor with shape (C, H, W) or a numpy array with shape (H, W, C)
    """
    use_torch_ops = isinstance(image, torch.Tensor)
    input_height, input_width = image.shape[1:] if use_torch_ops else image.shape[:2]
    if width == input_width and height == input_height:
        return image, (1.0, 1.0), (height, width)
    if force:
        return (
            tvtf.resize(image, size=[height, width])
            if use_torch_ops
            else cv2.resize(image, dsize=(width, height)),
            (height / input_height, width / input_width),
            (height, width),
        )
    scale = min(width / input_width, height / input_height)
    target_height = int(scale * input_height)
    target_width = int(scale * input_width)
    pad_y = height - target_height
    pad_x = width - target_width
    resized = (
        tvtf.resize(image, size=[target_height, target_width])
        if use_torch_ops
        else cv2.resize(image, (target_width, target_height))
    )
    if pad_y > 0 or pad_x > 0:
        padded = (
            torch.nn.functional.pad(resized, (0, pad_x, 0, pad_y))
            if use_torch_ops
            else np.pad(resized, ((0, pad_y), (0, pad_x), (0, 0)))
        )
    else:
        padded = resized
    return padded, (scale, scale), (target_height, target_width)


def resize(
    x: typing.List[ArrayType], resize_config: ResizeConfig
) -> typing.Tuple[ArrayType, ArrayType, ArrayType]:
    """Resize a list of images using a specified method.

    Args:
        x: A list of tensors of shape (C, H, W) or a tensor of
           shape (N, C, H, W) or a list of ndarrays of shape (H, W, C)
           or an ndarray of shape (N, H, W, C).
        resize_config: A resize config object.
    """
    assert (
        not isinstance(x, list) or len(x) > 0
    ), "When providing a list, it must not be empty."
    assert resize_config["method"] in [
        "fit",
        "pad",
        "none",
        "force",
        "pad_to_multiple",
    ], f"Unknown resize method {resize_config['method']}."
    use_torch_ops = isinstance(x, torch.Tensor) or (
        isinstance(x, list) and isinstance(x[0], torch.Tensor)
    )
    width, height, base = typing.cast(
        typing.List[typing.Optional[int]],
        [resize_config.get(k) for k in ["width", "height", "base"]],
    )
    if resize_config["method"] in ["fit", "force"]:
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using fit or force."
        resized_list, scale_list, size_list = zip(
            *[
                fit(
                    image,
                    height=height,
                    width=width,
                    force=resize_config["method"] == "force",
                )
                for image in x
            ]
        )
        return (
            (  # type: ignore
                torch.cat([r.unsqueeze(0) for r in resized_list]),
                torch.tensor(scale_list),
                torch.tensor(size_list),
            )
            if use_torch_ops
            else (
                np.concatenate([r[np.newaxis] for r in resized_list]),
                np.array(scale_list),
                np.array(size_list),
            )
        )
    img_dimensions = np.array(
        [i.shape[1:3] if use_torch_ops else i.shape[:2] for i in x]
    )
    scales = (
        torch.tensor(np.ones_like(img_dimensions))
        if use_torch_ops
        else np.ones_like(img_dimensions)
    )
    sizes = torch.tensor(img_dimensions) if use_torch_ops else np.array(img_dimensions)
    if resize_config["method"] == "pad":
        assert (
            height is not None and width is not None
        ), "You must provide width and height when using pad."
        pad_dimensions = np.array([[height, width]]) - img_dimensions
    if resize_config["method"] == "pad_to_multiple":
        assert base is not None, "pad_to_multiple requires a base to be provided."
        pad_dimensions = (
            (np.ceil(img_dimensions.max(axis=0) / base) * base)
            .clip(base)
            .astype("int32")
        ) - img_dimensions
    if resize_config["method"] == "none":
        # pylint: disable=unexpected-keyword-arg
        pad_dimensions = img_dimensions.max(axis=0, keepdims=True) - img_dimensions
    assert (
        pad_dimensions >= 0
    ).all(), "Input image is larger than target size, but method is 'pad.'"
    padded = (
        torch.cat(
            [
                torch.nn.functional.pad(i, (0, pad_x, 0, pad_y)).unsqueeze(0)  # type: ignore
                for i, (pad_y, pad_x) in zip(x, pad_dimensions)
            ]
        )
        if use_torch_ops
        else np.concatenate(
            [
                np.pad(i, ((0, pad_y), (0, pad_x), (0, 0)))[np.newaxis]
                for i, (pad_y, pad_x) in zip(x, pad_dimensions)
            ],
            axis=0,
        )
    )
    return padded, scales, sizes  # type: ignore


class BaseModel:
    """Abstract base class for classifiers and detectors."""

    model: torch.nn.Module
    backbone: torch.nn.Module
    annotation_config: annotation.AnnotationConfiguration
    device: typing.Any
    resize_config: ResizeConfig

    def resize_to_model_size(
        self, images: typing.List[np.ndarray]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Resize a series of images to the current model's size."""
        padded, scales, _ = resize(images, self.resize_config)
        return padded, scales

    def set_device(self, device):
        """Set the device for training and inference tasks."""
        self.device = torch.device(device)
        self.model.to(self.device)

    def compute_inputs(self, images: np.ndarray) -> torch.Tensor:
        """Compute the model inputs given a numpy array of images."""
        images = images.astype("float32") / 255.0
        return (
            torch.tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

    def load_weights(self, filepath: str):
        """Load weights from disk."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def save_weights(self, filepath: str):
        """Save weights to disk."""
        torch.save(self.model.state_dict(), filepath)

    def freeze_backbone(self):
        """Freeze the body of the model, leaving the final classification and
        regression layer as trainable."""
        for p in self.backbone.parameters():  # type: ignore
            p.requires_grad = False
        for m in self.backbone.modules():  # type: ignore
            m.eval()

    def unfreeze_backbone(self, batchnorm=True):
        """Unfreeze the body of the model, making all layers trainable.

        Args:
            batchnorm: Whether to unfreeze batchnorm layers.
        """
        for m in self.backbone.modules():  # type: ignore
            if isinstance(m, torch.nn.BatchNorm2d) and not batchnorm:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            else:
                m.train()
                for p in m.parameters():
                    p.requires_grad = True

    def n_parameters(self, trainable_only=False):
        """Count the number of model parameters."""
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad or not trainable_only
        )
