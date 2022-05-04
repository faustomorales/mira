# pylint: disable=too-many-public-methods
import os
import abc
import json
import types
import typing
import random
import logging
import tempfile

import cv2
import torch
import tqdm
import numpy as np
import timm.optim
import timm.scheduler
import pkg_resources
import typing_extensions as tx

try:
    import model_archiver.model_packaging as marmp
    import model_archiver.model_packaging_utils as marmpu
except ImportError:
    marmp = None
    marmpu = None

from .. import metrics as mm
from .. import core as mc
from . import common as mdc

DEFAULT_SCHEDULER_PARAMS = dict(
    sched="cosine",
    min_lr=1e-5,
    decay_rate=1,
    warmup_lr=0,
    warmup_epochs=0,
    cooldown_epochs=0,
    epochs=100,
    lr_cycle_limit=0,
)

DEFAULT_OPTIMIZER_PARAMS = dict(learning_rate=1e-2, weight_decay=4e-5)

ResizeMethod = tx.Literal["fit", "pad"]

LOGGER = logging.getLogger(__name__)


class Detector:
    """Abstract base class for a detector."""

    @property
    @abc.abstractmethod
    def anchor_boxes(self) -> np.ndarray:
        """Return the list of anchor boxes in xyxy format."""

    model: torch.nn.Module
    backbone: torch.nn.Module
    annotation_config: mc.AnnotationConfiguration
    training_model: torch.nn.Module
    device: typing.Any
    resize_method: tx.Literal["pad", "fit", "pad_to_multiple"]
    resize_base: typing.Optional[int]

    def set_device(self, device):
        """Set the device for training and inference tasks."""
        self.device = torch.device(device)
        self.model.to(self.device)
        if hasattr(self, "training_model"):
            self.training_model.to(self.device)  # type: ignore

    @abc.abstractmethod
    def invert_targets(
        self,
        y: typing.Any,
        threshold: float = 0.5,
        **kwargs,
    ) -> typing.List[typing.List[mc.Annotation]]:
        """Compute a list of annotation groups from model output."""

    def resize_to_model_size(
        self, images: typing.List[np.ndarray]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Resize a series of images to the current model's size."""
        height, width = self.input_shape[:2]
        padded, scales, _ = mdc.resize(
            images,
            resize_method=self.resize_method,
            height=height,
            width=width,
            base=getattr(self, "resize_base", None),
        )
        # Ensure that scaling maintains aspect ratio.
        np.testing.assert_allclose(*scales.T)
        return padded, scales[:, 0]

    @property
    @abc.abstractmethod
    def input_shape(self) -> typing.Tuple[int, int, int]:
        """Obtain the input shape for this model."""

    @abc.abstractmethod
    def set_input_shape(self, width: int, height: int):
        """Set the input shape for this model."""

    @abc.abstractmethod
    def serve_module_string(self, enable_flexible_size: bool) -> str:
        """Return the module string used as part of TorchServe."""

    @property
    @abc.abstractmethod
    def serve_module_index(self) -> dict:
        """Return the class index -> label mapping for TorchServe."""

    @abc.abstractmethod
    def compute_inputs(self, images: np.ndarray) -> np.ndarray:
        """Convert images into suitable model inputs. *You
        usually should not need this method*. For training,
        use `detector.train()`. For detection, use
        `detector.detect()`.

        Args:
            images: The images to convert

        Returns:
            The input to the model
        """

    @abc.abstractmethod
    def compute_targets(
        self,
        annotation_groups: typing.List[typing.List[mc.Annotation]],
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

    def loss_for_batch(
        self,
        batch,
        data_dir: str = None,
        transforms: np.ndarray = None,
        indices: typing.List[int] = None,
    ):
        """Compute the loss for a batch of scenes."""
        assert self.training_model.training, "Model not in training mode."
        images, scales = self.resize_to_model_size(batch.images)
        annotation_groups = [
            [ann.resize(scale) for ann in scene.annotations]
            for scene, scale in zip(batch, scales)
        ]
        output = self.training_model(
            self.compute_inputs(images),
            self.compute_targets(
                annotation_groups=annotation_groups,
                width=images.shape[2],
                height=images.shape[1],
            ),
        )
        if data_dir is not None:
            assert transforms is not None, "Transforms are required for data caching."
            assert indices is not None, "Image indices are required for data caching."
            os.makedirs(data_dir, exist_ok=True)
            for outputIdx, (idx, image, anns, transform, scale, metadata) in enumerate(
                zip(
                    indices or np.arange(len(images)),
                    images,
                    annotation_groups,
                    transforms,
                    scales,
                    [s.metadata for s in batch],
                )
            ):
                assert cv2.imwrite(
                    os.path.join(data_dir, str(idx) + ".png"), image[..., ::-1]
                )

                with open(
                    os.path.join(data_dir, str(idx) + ".png.metadata.json"),
                    "w",
                    encoding="utf8",
                ) as f:
                    f.write(json.dumps(metadata or {}))
                np.savez(
                    os.path.join(data_dir, str(idx) + ".png.output.npz"),
                    **{
                        k: v.detach().cpu()
                        for k, v in output["output"][outputIdx].items()
                    },
                )
                np.savez(
                    os.path.join(data_dir, str(idx) + ".png.bboxes.npz"),
                    bboxes=batch.annotation_config.bboxes_from_group(anns),
                )
                np.savez(
                    os.path.join(data_dir, str(idx) + ".png.transform.npz"),
                    transform=np.matmul(
                        np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]), transform
                    ),
                )
        return output["loss"]

    def train(
        self,
        training: mc.SceneCollection,
        validation: mc.SceneCollection = None,
        batch_size: int = 1,
        augmenter: mc.augmentations.AugmenterProtocol = None,
        train_backbone: bool = True,
        train_backbone_bn: bool = True,
        epochs=100,
        shuffle=True,
        callbacks: typing.List[mc.callbacks.CallbackProtocol] = None,
        optimizer_params=None,
        scheduler_params=None,
        clip_grad_norm_params=None,
        data_dir_prefix=None,
        validation_transforms: np.ndarray = None,
    ):
        """Run training job.
        Args:
            training: The collection of training images
            validation: The collection of validation images
            batch_size: The batch size to use for training
            augmenter: The augmenter for generating samples
            train_backbone: Whether to fit the backbone.
            epochs: The number of epochs to train.
            shuffle: Whether to shuffle the training data on each epoch.
            callbacks: A list of functions that accept the detector as well
                as a list of previous summaries and returns a dict of summary keys
                and values which will be added to the current summary. It can raise
                StopIteration to stop training early.
            optimizer_params: Passed to timm.optim.create_optimizer_v2 to build
                the optimizer.
            scheduler_params: Passed to timm.schduler.create_scheduler to build
                the scheduler.
        """
        training_model = (
            self.training_model if hasattr(self, "training_model") else self.model
        )
        assert training_model is not None
        optimizer = timm.optim.create_optimizer_v2(
            training_model, **(optimizer_params or DEFAULT_OPTIMIZER_PARAMS)
        )
        scheduler, _ = timm.scheduler.create_scheduler(
            types.SimpleNamespace(**(scheduler_params or DEFAULT_SCHEDULER_PARAMS)),
            optimizer=optimizer,
        )
        train_index = np.arange(len(training)).tolist()
        if validation is not None and validation_transforms is None:
            LOGGER.warning(
                "No validation transforms were provided. Assuming identity matrix."
            )
            validation_transforms = np.eye(3, 3)[np.newaxis].repeat(
                len(validation), axis=0
            )
        summaries: typing.List[typing.Dict[str, typing.Any]] = []
        for epoch in range(epochs):
            with tqdm.trange(
                len(training) // batch_size
            ) as t, tempfile.TemporaryDirectory(prefix=data_dir_prefix) as tdir:
                training_model.train()
                if not train_backbone:
                    self.freeze_backbone()
                else:
                    self.unfreeze_backbone(batchnorm=train_backbone_bn)
                t.set_description(f"Epoch {epoch + 1} / {epochs}")
                scheduler.step(
                    epoch=epoch,
                    metric=None
                    if not summaries
                    else summaries[-1].get("val_loss", summaries[-1]["loss"]),
                )
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    end = min(start + batch_size, len(train_index))
                    train_indices = [train_index[idx] for idx in range(start, end)]
                    batch = training.assign(
                        scenes=[training[idx] for idx in train_indices]
                    )
                    if augmenter is not None:
                        batch, transforms = batch.augment(augmenter=augmenter)
                    else:
                        transforms = np.eye(3, 3)[np.newaxis].repeat(len(batch), axis=0)
                    optimizer.zero_grad()
                    loss = self.loss_for_batch(
                        batch,
                        data_dir=os.path.join(tdir, "train", str(batchIdx)),
                        transforms=transforms,
                        indices=train_indices,
                    )
                    loss.backward()
                    if clip_grad_norm_params is not None:
                        torch.nn.utils.clip_grad_norm_(
                            training_model.parameters(), **clip_grad_norm_params
                        )
                    cum_loss += loss.detach().cpu().numpy()
                    avg_loss = cum_loss / end
                    optimizer.step()
                    t.set_postfix(loss=avg_loss)
                    t.update()
                summary: typing.Dict[str, typing.Any] = {"loss": avg_loss}
                if validation is not None:
                    summary["val_loss"] = np.sum(
                        [
                            self.loss_for_batch(
                                validation.assign(
                                    scenes=[
                                        validation[idx]
                                        for idx in range(
                                            vstart,
                                            min(vstart + batch_size, len(validation)),
                                        )
                                    ]
                                ),
                                data_dir=os.path.join(tdir, "val", str(batchIdx)),
                                transforms=validation_transforms,
                                indices=np.arange(
                                    start=vstart, stop=vstart + batch_size
                                ).tolist(),
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            for batchIdx, vstart in enumerate(
                                range(0, len(validation), batch_size)
                            )
                        ]
                    ) / len(validation)
                summary["lr"] = next(g["lr"] for g in optimizer.param_groups)
                if callbacks is not None:
                    try:
                        for callback in callbacks:
                            for k, v in callback(
                                detector=self,
                                summaries=summaries + [summary],
                                data_dir=tdir,
                            ).items():
                                summary[k] = v
                    except StopIteration:
                        return summaries
                t.set_postfix(**summary)
                summaries.append(summary)
        return summaries

    def detect(self, image: np.ndarray, **kwargs) -> typing.List[mc.Annotation]:
        """Run detection for a given image. All other args passed to invert_targets()

        Args:
            image: The image to run detection on

        Returns:
            A list of annotations
        """
        self.model.eval()
        images, scales = self.resize_to_model_size([image])
        with torch.no_grad():
            annotations = self.invert_targets(
                self.model(self.compute_inputs(images)), **kwargs
            )[0]
        return [a.resize(1 / scales[0]) for a in annotations]

    def detect_batch(
        self,
        images: typing.List[np.ndarray],
        batch_size: int = 32,
        **kwargs,
    ) -> typing.List[typing.List[mc.Annotation]]:
        """
        Perform object detection on a batch of images.

        Args:
            images: A list of images
            threshold: The detection threshold for the images
            batch_size: The batch size to use with the underlying model

        Returns:
            A list of lists of annotations.
        """
        self.model.eval()
        annotation_groups = []
        for start in range(0, len(images), batch_size):
            current_images, current_scales = self.resize_to_model_size(
                images[start : start + batch_size]
            )
            annotation_groups.extend(
                [
                    [a.resize(1 / scale) for a in annotations]
                    for scale, annotations in zip(
                        current_scales,
                        self.invert_targets(
                            self.model(
                                self.compute_inputs(current_images),
                            ),
                            **kwargs,
                        ),
                    )
                ]
            )
        return annotation_groups

    def mAP(
        self,
        collection: mc.SceneCollection,
        iou_threshold=0.5,
        min_threshold=0.01,
        batch_size=32,
    ):
        """Compute the mAP metric for a given collection
        of ground truth scenes.

        Args:
            collection: The collection to evaluate
            min_threshold: The minimum threshold for initial selection
                of boxes.
            iou_threshold: The IoU threshold required for
                a match

        Returns:
            mAP score
        """
        pred = collection.assign(
            scenes=[
                scene.assign(annotations=annotations)
                for scene, annotations in zip(
                    collection,
                    self.detect_batch(
                        images=collection.images,
                        threshold=min_threshold,
                        batch_size=batch_size,
                    ),
                )
            ]
        )
        return mm.mAP(
            true_collection=collection,
            pred_collection=pred,
            iou_threshold=iou_threshold,
        )

    def to_torchserve(
        self,
        model_name: str,
        directory=".",
        archive_format: tx.Literal["default", "no-archive"] = "default",
        score_threshold: float = 0.5,
        enable_flexible_size=False,
        model_version="1.0",
    ):
        """Build a TorchServe-compatible MAR file for this model."""
        assert (
            marmpu is not None
        ), "You must `pip install torch-model-archiver` to use this function."
        os.makedirs(directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as tdir:
            serialized_file = os.path.join(tdir, "weights.pth")
            index_to_name_file = os.path.join(tdir, "index_to_name.json")
            model_file = os.path.join(tdir, "model.py")
            handler_file = os.path.join(tdir, "object_detector.py")
            torch.save(self.model.state_dict(prefix="model."), serialized_file)
            with open(index_to_name_file, "w", encoding="utf8") as f:
                f.write(json.dumps(self.serve_module_index))
            with open(model_file, "w", encoding="utf8") as f:
                f.write(
                    self.serve_module_string(enable_flexible_size=enable_flexible_size)
                )
            with open(handler_file, "w", encoding="utf8") as f:
                f.write(
                    pkg_resources.resource_string(
                        "mira", "detectors/assets/serve/object_detector.py"
                    )
                    .decode("utf-8")
                    .replace("SCORE_THRESHOLD", str(score_threshold))  # type: ignore
                )
            args = types.SimpleNamespace(
                model_name=model_name,
                serialized_file=serialized_file,
                handler=handler_file,
                model_file=model_file,
                version=model_version,
                requirements_file=None,
                runtime="python",
                extra_files=index_to_name_file,
                export_path=directory,
                force=True,
                archive_format=archive_format,
            )
            marmp.package_model(
                args=args, manifest=marmpu.ModelExportUtils.generate_manifest_json(args)
            )

    @property
    def anchor_sizes(self):
        """Get an array of anchor sizes (i.e., width and height)."""
        return np.diff(
            self.anchor_boxes.reshape((-1, 2, 2)),
            axis=1,
        )[:, 0, :]
