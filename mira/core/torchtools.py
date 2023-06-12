import os
import abc
import glob
import json
import math
import types
import typing
import random
import logging
import warnings
import tempfile


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
import tqdm
import numpy as np
import pandas as pd
import typing_extensions as tx
from . import annotation, resizing, scene, augmentations, utils

warnings.filterwarnings(
    "ignore",
    message="The epoch parameter.*",
)

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
TrainItem = typing.NamedTuple(
    "TrainItem",
    [
        ("index", int),
        ("transform", np.ndarray),
        ("scene", scene.Scene),
    ],
)
TrainState = tx.TypedDict("TrainState", {"directory": tempfile.TemporaryDirectory})
InvertedTarget = typing.NamedTuple(
    "InvertedTarget",
    [
        ("labels", typing.List[annotation.Label]),
        ("annotations", typing.List[annotation.Annotation]),
    ],
)
CollectionPair = tx.TypedDict(
    "CollectionPair",
    {
        "true_collection": scene.SceneCollection,
        "pred_collection": scene.SceneCollection,
    },
)
SplitCollection = tx.TypedDict(
    "SplitCollection",
    {
        "collections": CollectionPair,
        "transforms": np.ndarray,
        "metadata": pd.DataFrame,
        "indices": typing.List[int],
    },
)


# pylint: disable=too-few-public-methods
class CallbackProtocol(tx.Protocol):
    """A protocol defining how we expect callbacks to behave."""

    def __call__(
        self,
        model: "BaseModel",
        summaries: typing.List[typing.Dict[str, typing.Any]],
        collections: typing.Dict[str, SplitCollection],
    ) -> typing.Dict[str, typing.Any]:
        pass


def train(
    model: "torch.nn.Module",
    loss: typing.Callable[
        [tx.Literal["training", "validation"], typing.List[InputType]], "torch.Tensor"
    ],
    training: typing.List[InputType],
    skip_partial_batches=False,
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
    optimizer_params = optimizer_params or DEFAULT_OPTIMIZER_PARAMS
    if "model_or_params" not in optimizer_params:
        optimizer_params = {**optimizer_params, "model_or_params": model}
    optimizer = timm.optim.create_optimizer_v2(**optimizer_params)
    scheduler_params = scheduler_params or DEFAULT_SCHEDULER_PARAMS
    scheduler, _ = timm.scheduler.create_scheduler(
        types.SimpleNamespace(**scheduler_params),
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
                    if not summaries or "eval_metric" not in scheduler_params
                    else summaries[-1][scheduler_params["eval_metric"]],
                )
                if on_epoch_start:
                    on_epoch_start()
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    end = min(start + batch_size, len(train_index))
                    batch = [training[train_index[idx]] for idx in range(start, end)]
                    if len(batch) < batch_size and skip_partial_batches:
                        continue
                    if augment:
                        batch = augment(batch)
                    optimizer.zero_grad()
                    batch_loss = loss("training", batch)
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
                                "validation",
                                [
                                    validation[idx]
                                    for idx in range(
                                        vstart,
                                        min(vstart + batch_size, len(validation)),
                                    )
                                ],
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            for vstart in range(0, len(validation), batch_size)
                            if not skip_partial_batches
                            or len(validation) - vstart >= batch_size
                        ]
                    ) / len(validation)
                summaries[-1]["lr"] = max(g["lr"] for g in optimizer.param_groups)
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


BatchInferenceItem = typing.Union[
    typing.List[typing.Union[np.ndarray, typing.Callable[[], np.ndarray]]],
    np.ndarray,
    scene.SceneCollection,
    scene.Scene,
]
BatchInferenceOutput = typing.TypeVar("BatchInferenceOutput")
BatchInferenceInput = typing.NamedTuple(
    "BatchInferenceInput", [("images", np.ndarray), ("scales", np.ndarray)]
)


def data_dir_to_collections(
    data_dir: str, threshold: float, model: "BaseModel"
) -> typing.Dict[str, SplitCollection]:
    """Convert a temporary training artifact directory into a set
    of train and validation (if present) true/predicted collections."""
    return {
        split: {
            "collections": {
                "true_collection": scene.SceneCollection(
                    [
                        scene.Scene(
                            image=filepath,
                            categories=model.categories,
                            annotations=[
                                annotation.Annotation(
                                    model.categories[cIdx], x1, y1, x2, y2
                                )
                                for x1, y1, x2, y2, cIdx in np.load(
                                    filepath + ".bboxes.npz"
                                )["bboxes"]
                            ],
                            labels=[
                                annotation.Label(category=model.categories[cIdx])
                                for cIdx, value in enumerate(
                                    np.load(filepath + ".labels.npz")["labels"]
                                )
                                if value != 0
                            ],
                            metadata=metadata,
                        )
                        for filepath, metadata in zip(images, metadatas)
                    ],
                    categories=model.categories,
                ),
                "pred_collection": scene.SceneCollection(
                    [
                        scene.Scene(
                            image=filepath,
                            categories=model.categories,
                            annotations=inverted.annotations,
                            labels=inverted.labels,
                            metadata=metadata,
                        )
                        for filepath, metadata, inverted in zip(
                            images,
                            metadatas,
                            model.invert_targets(
                                {
                                    "output": [
                                        {
                                            k: torch.Tensor(v)
                                            for k, v in np.load(
                                                filepath + ".output.npz"
                                            ).items()
                                        }
                                        for filepath in images
                                    ]
                                },
                                threshold=threshold,
                            ),
                        )
                    ],
                    categories=model.categories,
                ),
            },
            "transforms": typing.cast(np.ndarray, transforms),
            "metadata": pd.json_normalize(metadatas),
            "indices": [
                int(os.path.basename(filepath).split(".")[0]) for filepath in images
            ],
        }
        for split, images, metadatas, transforms in [
            (
                split,
                images,
                [utils.load_json(f + ".metadata.json") for f in images],
                [np.load(f + ".transform.npz")["transform"] for f in images],
            )
            for split, images in [
                (
                    split,
                    glob.glob(os.path.join(data_dir, split, "*.png"))
                    + glob.glob(os.path.join(data_dir, split, "*", "*.png")),
                )
                for split in ["training", "validation"]
            ]
        ]
        if len(images) > 0
    }


class BaseModel:
    """Abstract base class for classifiers and detectors."""

    model: "torch.nn.Module"
    backbone: "torch.nn.Module"
    categories: annotation.Categories
    device: typing.Any
    resize_config: resizing.ResizeConfig

    @abc.abstractmethod
    def compute_targets(
        self,
        targets: typing.List[InvertedTarget],
        width: int,
        height: int,
    ) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
        """Compute the expected outputs for a model. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            annotation_groups: A list of lists of annotation groups.

        Returns:
            The output(s) that will be used by detector.train()
        """

    @abc.abstractmethod
    def invert_targets(
        self, y: typing.Any, threshold: float
    ) -> typing.List[InvertedTarget]:
        """Convert model outputs back into predictions."""

    def resize_to_model_size(
        self, images: typing.List[np.ndarray]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Resize a series of images to the current model's size."""
        padded, scales, _ = resizing.resize(images, self.resize_config)
        return padded, scales

    def set_device(self, device):
        """Set the device for training and inference tasks."""
        self.device = torch.device(device)
        self.model.to(self.device)

    def compute_inputs(self, images: np.ndarray) -> "torch.Tensor":
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

    def batch_inference(
        self,
        items: BatchInferenceItem,
        process: typing.Callable[
            [BatchInferenceInput],
            typing.List[BatchInferenceOutput],
        ],
        batch_size: int,
        progress: bool,
    ) -> typing.Tuple[bool, typing.List[BatchInferenceOutput]]:
        """Given some set of items, which could be a scene collection or a scene
        or a list of images or deferred images (pretty much anything), batch it
        into (images, scales) tuples."""
        single = False
        images: typing.Sequence[
            typing.Union[np.ndarray, str, typing.Callable[[], np.ndarray]]
        ]
        if isinstance(items, scene.SceneCollection):
            images = items.deferred_images()
        elif isinstance(items, scene.Scene):
            single = True
            images = [items.image]
        else:
            single = (
                isinstance(items, np.ndarray)
                and len(typing.cast(np.ndarray, items).shape) == 3
            )
            images = typing.cast(
                typing.List[typing.Union[np.ndarray, str]], [items] if single else items
            )
        self.model.eval()

        iterator = range(0, len(images), batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator, total=math.ceil(len(images) / batch_size))
        accumulated = []
        with torch.no_grad():
            for start in iterator:
                accumulated.extend(
                    process(
                        BatchInferenceInput(
                            *self.resize_to_model_size(
                                [
                                    image
                                    if isinstance(image, np.ndarray)
                                    else (
                                        utils.read(typing.cast(str, image))
                                        if isinstance(image, str)
                                        else image()
                                    )
                                    for image in images[start : start + batch_size]
                                ]
                            )
                        )
                    )
                )

        return single, accumulated

    def loss(
        self,
        batch: scene.SceneCollection,
        data_dir: str = None,
        transforms: np.ndarray = None,
        indices: typing.List[int] = None,
        save_images=False,
    ) -> "torch.Tensor":
        """Compute the loss for a batch of scenes."""
        assert self.model.training, "Model not in training mode."
        images, scales = self.resize_to_model_size(batch.images())
        LOGGER.debug(
            "Obtained images array with size %s and scales varying from %s to %s",
            images.shape,
            scales.min(),
            scales.max(),
        )
        itargets = [
            InvertedTarget(
                annotations=[ann.resize(scale) for ann in scene.annotations],
                labels=scene.labels,
            )
            for scene, scale in zip(batch, scales[:, ::-1])
        ]
        output = self.model(
            self.compute_inputs(images),
            self.compute_targets(
                targets=itargets,
                width=images.shape[2],
                height=images.shape[1],
            ),
        )
        if data_dir is not None:
            assert transforms is not None, "Transforms are required for data caching."
            assert indices is not None, "Image indices are required for data caching."
            os.makedirs(data_dir, exist_ok=True)
            for outputIdx, (
                idx,
                image,
                itarget,
                lbls,
                transform,
                (scaley, scalex),
                metadata,
            ) in enumerate(
                zip(
                    indices,
                    images,
                    itargets,
                    batch.onehot(),
                    transforms,
                    scales,
                    [s.metadata for s in batch],
                )
            ):
                base_path = os.path.join(data_dir, str(idx))
                assert cv2.imwrite(
                    base_path + ".png",
                    image[..., ::-1]
                    if save_images
                    else np.ones((1, 1, 3), dtype="uint8"),
                )

                with open(
                    base_path + ".png.metadata.json",
                    "w",
                    encoding="utf8",
                ) as f:
                    f.write(json.dumps(metadata or {}))
                np.savez(
                    base_path + ".png.output.npz",
                    **{
                        k: v.detach().cpu()
                        for k, v in output["output"][outputIdx].items()
                    },
                )
                np.savez(
                    base_path + ".png.bboxes.npz",
                    bboxes=batch.categories.bboxes_from_group(itarget.annotations),
                )
                np.savez(
                    base_path + ".png.labels.npz",
                    labels=lbls,
                )
                np.savez(
                    base_path + ".png.transform.npz",
                    transform=np.matmul(
                        np.array([[scalex, 0, 0], [0, scaley, 0], [0, 0, 1]]),
                        transform,
                    ),
                )
        return output["loss"]

    # pylint: disable=consider-using-with
    def train(
        self,
        training: scene.SceneCollection,
        validation: scene.SceneCollection = None,
        augmenter: augmentations.AugmenterProtocol = None,
        train_backbone: bool = True,
        train_backbone_bn: bool = True,
        callbacks: typing.List[CallbackProtocol] = None,
        data_dir_prefix=None,
        validation_transforms: np.ndarray = None,
        min_visibility: float = None,
        save_images=True,
        **kwargs,
    ):
        """Run training job. All other arguments passed to mira.core.torchtools.train.

        Args:
            training: The collection of training images
            validation: The collection of validation images
            augmenter: The augmenter for generating samples
            train_backbone: Whether to fit the backbone.
            train_backbone_bn: Whether to fit backbone batchnorm layers.
            callbacks: A list of training callbacks.
            data_dir_prefix: Prefix for the intermediate model artifacts directory.
            validation_transforms: A list of transforms for the images in the validation
                set. If not provided, we assume the identity transform.
        """
        state: TrainState = {
            "directory": tempfile.TemporaryDirectory(prefix=data_dir_prefix),
        }

        def loss(split: str, items: typing.List[TrainItem]) -> "torch.Tensor":
            return self.loss(
                training.assign(scenes=[i.scene for i in items]),
                data_dir=os.path.join(state["directory"].name, split),
                transforms=np.stack([i.transform for i in items]),
                indices=[i.index for i in items],
                save_images=save_images,
            )

        def augment(items: typing.List[TrainItem]):
            if not augmenter:
                return items
            return [
                TrainItem(
                    index=base.index,
                    scene=scene,
                    transform=np.matmul(transform, base.transform),
                )
                for (scene, transform), base in zip(
                    [
                        i.scene.augment(augmenter, min_visibility=min_visibility)
                        for i in items
                    ],
                    items,
                )
            ]

        def on_epoch_end(summaries: typing.List[dict]):
            summary: typing.Dict[str, typing.Any] = summaries[-1]
            collections = data_dir_to_collections(
                data_dir=state["directory"].name, threshold=0.01, model=self
            )
            if callbacks:
                for callback in callbacks:
                    for k, v in callback(
                        model=self,
                        summaries=summaries,
                        collections=collections,
                    ).items():
                        summary[k] = v
            state["directory"] = tempfile.TemporaryDirectory(prefix=data_dir_prefix)
            return summary

        def on_epoch_start():
            if train_backbone:
                self.unfreeze_backbone(batchnorm=train_backbone_bn)
            else:
                self.freeze_backbone()

        return train(
            model=self.model,
            training=[
                TrainItem(index=index, transform=np.eye(3), scene=scene)
                for index, scene in enumerate(training)
            ],
            validation=[
                TrainItem(index=index, transform=transform, scene=scene)
                for index, (scene, transform) in enumerate(
                    zip(
                        validation or [],
                        (
                            validation_transforms
                            or np.eye(3, 3)[np.newaxis].repeat(len(validation), axis=0)
                        )
                        if validation
                        else [],
                    )
                )
            ],
            loss=loss,
            augment=augment,
            on_epoch_start=on_epoch_start,
            on_epoch_end=on_epoch_end,
            **kwargs,
        )


def get_linear_lr_scales(model, frozen=0, min_scale=None):
    """Build a timm-compatible array of parameter groups
    with a specific number or portion of parameters
    left frozen. The remaining parameters will have
    learning rate scales linearly increasing from
    0 to 1."""
    parameters = list(p for p in model.parameters())
    ngroups = len(parameters)
    if isinstance(frozen, float) and 0 <= frozen <= 1:
        frozen = int(round(frozen * ngroups))
    ngroups = ngroups - frozen
    if min_scale is None:
        min_scale = 1 / ngroups
    slope = (1 - min_scale) / ngroups
    return [
        {
            "params": p,
            "lr_scale": 0 if idx < frozen else slope * (idx - frozen + 1) + min_scale,
        }
        for idx, p in enumerate(parameters)
    ]


def logits2labels(
    logits: "torch.Tensor", categories: annotation.Categories, threshold: float
):
    """Convert a batch of logits to category labels."""
    scores = logits.softmax(dim=-1).numpy()
    return [
        InvertedTarget(
            labels=[
                annotation.Label(
                    category=categories[classIdx],
                    score=score,
                    metadata={
                        "logit": logit,
                        "raw": {
                            category.name: {"score": score, "logit": logit}
                            for category, score, logit in zip(
                                categories,
                                catscores.tolist(),
                                catlogits.tolist(),
                            )
                        },
                    },
                )
            ]
            if score >= threshold
            else [],
            annotations=[],
        )
        for score, classIdx, logit, catscores, catlogits in zip(
            scores.max(axis=1).tolist(),
            scores.argmax(axis=1).tolist(),
            logits.numpy().max(axis=1).tolist(),
            scores,
            logits.numpy(),
        )
    ]
