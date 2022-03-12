# pylint: disable=too-many-public-methods
import os
import abc
import json
import types
import typing
import random
import tempfile

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

DEFAULT_SCHEDULER_PARAMS = dict(
    sched="cosine",
    min_lr=1e-5,
    decay_rate=0.1,
    warmup_lr=1e-4,
    warmup_epochs=5,
    cooldown_epochs=10,
)

DEFAULT_OPTIMIZER_PARAMS = dict(learning_rate=1e-2, weight_decay=4e-5)

ResizeMethod = tx.Literal["fit", "pad"]


def _loss_from_loss_dict(loss_dict: typing.Dict[str, torch.Tensor]):
    if "loss" in loss_dict:
        return loss_dict["loss"]
    return sum(loss for loss in loss_dict.values())


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
    resize_method: tx.Literal["pad", "fit"]

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
        self, image: np.ndarray
    ) -> typing.Tuple[np.ndarray, float]:
        """Resize image to model size."""
        if self.resize_method == "fit":
            return self.fit_to_model_size(image)
        if self.resize_method == "pad":
            return self.pad_to_model_size(image)
        raise NotImplementedError(f"Unknown resize method: {self.resize_method}")

    def fit_to_model_size(self, image: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """Fit an image to model size by up-or-downsampling the constraining dimension
        and then padding the the other dimension to size."""
        height, width = self.input_shape[:2]
        image, scale = mc.utils.fit(image=image, width=width, height=height)
        return image, scale

    def pad_to_model_size(self, image: np.ndarray) -> typing.Tuple[np.ndarray, float]:
        """Pad images to model size."""
        height, width = self.input_shape[:2]
        assert image.shape[0] <= height, "Cannot pad image."
        assert image.shape[1] <= width, "Cannot pad image."
        padded = mc.utils.get_blank_image(
            width=width, height=height, n_channels=3, cval=0
        )
        padded[: image.shape[0], : image.shape[1]] = image
        return padded, 1

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
    def compute_inputs(self, images: typing.List[np.ndarray]) -> np.ndarray:
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

    def loss_for_batch(self, batch):
        """Compute the loss for a batch of scenes."""
        assert self.training_model.training, "Model not in training mode."
        images, scales = list(
            zip(*[self.resize_to_model_size(s.image) for s in batch.scenes])
        )
        annotation_groups = [
            [ann.resize(scale) for ann in scene.annotations]
            for scene, scale in zip(batch, scales)
        ]
        return _loss_from_loss_dict(
            self.training_model(
                self.compute_inputs(images),
                self.compute_targets(annotation_groups=annotation_groups),
            )
        )

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
        scheduler, num_epochs = timm.scheduler.create_scheduler(
            types.SimpleNamespace(
                **{**(scheduler_params or DEFAULT_SCHEDULER_PARAMS), "epochs": epochs}
            ),
            optimizer=optimizer,
        )
        train_index = np.arange(len(training)).tolist()
        summaries = []
        for epoch in range(num_epochs):
            with tqdm.trange(len(training) // batch_size) as t:
                training_model.train()
                if not train_backbone:
                    self.freeze_backbone()
                else:
                    self.unfreeze_backbone(batchnorm=train_backbone_bn)
                t.set_description(f"Epoch {epoch + 1} / {num_epochs}")
                cum_loss = 0
                for batchIdx, start in enumerate(range(0, len(training), batch_size)):
                    if batchIdx == 0 and shuffle:
                        random.shuffle(train_index)
                    end = min(start + batch_size, len(train_index))
                    batch = training.assign(
                        scenes=[training[train_index[idx]] for idx in range(start, end)]
                    )
                    if augmenter is not None:
                        batch = batch.augment(augmenter=augmenter)
                    optimizer.zero_grad()
                    loss = self.loss_for_batch(batch)
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
                summaries.append(summary)
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
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            for vstart in range(0, len(validation), batch_size)
                        ]
                    ) / len(validation)
                scheduler.step(
                    epoch=epoch,
                    metric=avg_loss if validation is None else summary["val_loss"],
                )
                if callbacks is not None:
                    try:
                        for callback in callbacks:
                            for k, v in callback(
                                detector=self, summaries=summaries
                            ).items():
                                summary[k] = v
                    except StopIteration:
                        return summaries
                t.set_postfix(**summary)
        return summaries

    def detect(self, image: np.ndarray, **kwargs) -> typing.List[mc.Annotation]:
        """Run detection for a given image. All other args passed to invert_targets()

        Args:
            image: The image to run detection on

        Returns:
            A list of annotations
        """
        self.model.eval()
        image, scale = self.resize_to_model_size(image)
        with torch.no_grad():
            annotations = self.invert_targets(
                self.model(self.compute_inputs([image])), **kwargs
            )[0]
        return [a.resize(1 / scale) for a in annotations]

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
        images, scales = list(
            zip(*[self.resize_to_model_size(image) for image in images])
        )
        annotation_groups = []
        for start in range(0, len(images), batch_size):
            annotation_groups.extend(
                self.invert_targets(
                    self.model(
                        self.compute_inputs(images[start : start + batch_size]),
                    ),
                    **kwargs,
                )
            )
        return [
            [a.resize(1 / scale) for a in annotations]
            for annotations, scale in zip(annotation_groups, scales)
        ]

    def mAP(self, collection: mc.SceneCollection, iou_threshold=0.5, batch_size=32):
        """Compute the mAP metric for a given collection
        of ground truth scenes.

        Args:
            collection: The collection to evaluate
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
                        images=collection.images, threshold=0.01, batch_size=batch_size
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
