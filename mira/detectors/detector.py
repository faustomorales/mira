# pylint: disable=too-many-public-methods
import os
import abc
import json
import types
import typing
import logging
import tempfile

import cv2
import torch
import numpy as np
import pkg_resources
import typing_extensions as tx

try:
    import model_archiver.model_packaging as marmp
    import model_archiver.model_packaging_utils as marmpu
except ImportError:
    marmp, marmpu = None, None

from .. import metrics as mm
from .. import core as mc
from . import common as mdc

LOGGER = logging.getLogger(__name__)

TrainItem = typing.NamedTuple(
    "TrainItem",
    [
        ("split", tx.Literal["train", "val"]),
        ("index", int),
        ("transform", np.ndarray),
        ("scene", mc.Scene),
    ],
)
TrainState = tx.TypedDict("TrainState", {"directory": tempfile.TemporaryDirectory})


class Detector:
    """Abstract base class for a detector."""

    model: torch.nn.Module
    backbone: torch.nn.Module
    annotation_config: mc.AnnotationConfiguration
    device: typing.Any
    resize_config: mdc.ResizeConfig

    def set_device(self, device):
        """Set the device for training and inference tasks."""
        self.device = torch.device(device)
        self.model.to(self.device)

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
        padded, scales, _ = mdc.resize(images, self.resize_config)
        return padded, scales

    @abc.abstractmethod
    def serve_module_string(self) -> str:
        """Return the module string used as part of TorchServe."""

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

    @abc.abstractmethod
    def compute_anchor_boxes(self, width: int, height: int) -> np.ndarray:
        """Return the list of anchor boxes in xyxy format. You can convert these
        to dimensions using something like:

        detector.compute_anchor_boxes(iwidth, iheight)[:, [0, 2, 1, 3]].reshape((-1, 2, 2))
        """

    def loss(
        self,
        batch: mc.SceneCollection,
        data_dir: str = None,
        transforms: np.ndarray = None,
        indices: typing.List[int] = None,
    ) -> torch.Tensor:
        """Compute the loss for a batch of scenes."""
        assert self.model.training, "Model not in training mode."
        images, scales = self.resize_to_model_size(batch.images)
        LOGGER.debug(
            "Obtained images array with size %s and scales varying from %s to %s",
            images.shape,
            scales.min(),
            scales.max(),
        )
        annotation_groups = [
            [ann.resize(scale) for ann in scene.annotations]
            for scene, scale in zip(batch, scales[:, ::-1])
        ]
        output = self.model(
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
            for outputIdx, (
                idx,
                image,
                anns,
                transform,
                (scaley, scalex),
                metadata,
            ) in enumerate(
                zip(
                    indices,
                    images,
                    annotation_groups,
                    transforms,
                    scales,
                    [s.metadata for s in batch],
                )
            ):
                base_path = os.path.join(data_dir, str(idx))
                assert cv2.imwrite(base_path + ".png", image[..., ::-1])

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
                    bboxes=batch.annotation_config.bboxes_from_group(anns),
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
        training: mc.SceneCollection,
        validation: mc.SceneCollection = None,
        augmenter: mc.augmentations.AugmenterProtocol = None,
        train_backbone: bool = True,
        train_backbone_bn: bool = True,
        callbacks: typing.List[mc.callbacks.CallbackProtocol] = None,
        data_dir_prefix=None,
        validation_transforms: np.ndarray = None,
        **kwargs,
    ):
        """Run training job. All other arguments passed to mira.core.training.train.

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

        def loss(items: typing.List[TrainItem]) -> torch.Tensor:
            return self.loss(
                training.assign(scenes=[i.scene for i in items]),
                data_dir=os.path.join(state["directory"].name, items[0].split),
                transforms=np.stack([i.transform for i in items]),
                indices=[i.index for i in items],
            )

        def augment(items: typing.List[TrainItem]):
            if not augmenter:
                return items
            return [
                TrainItem(
                    split=base.split,
                    index=base.index,
                    scene=scene,
                    transform=np.matmul(transform, base.transform),
                )
                for (scene, transform), base in zip(
                    [i.scene.augment(augmenter) for i in items], items
                )
            ]

        def on_epoch_end(summaries: typing.List[dict]):
            summary: typing.Dict[str, typing.Any] = {}
            if callbacks:
                for callback in callbacks:
                    for k, v in callback(
                        detector=self,
                        summaries=summaries + [summary],
                        data_dir=state["directory"].name,
                    ).items():
                        summary[k] = v
            state["directory"] = tempfile.TemporaryDirectory(prefix=data_dir_prefix)
            return summary

        def on_epoch_start():
            if train_backbone:
                self.unfreeze_backbone(batchnorm=train_backbone_bn)
            else:
                self.freeze_backbone()

        mc.training.train(
            model=self.model,
            training=[
                TrainItem(split="train", index=index, transform=np.eye(3), scene=scene)
                for index, scene in enumerate(training)
            ],
            validation=[
                TrainItem(split="val", index=index, transform=transform, scene=scene)
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

    def detect(
        self,
        images: typing.Union[typing.List[np.ndarray], np.ndarray],
        batch_size: int = 32,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[mc.Annotation]], typing.List[mc.Annotation]
    ]:
        """
        Perform object detection on a batch of images or single image.

        Args:
            images: A list of images or a single image.
            threshold: The detection threshold for the images
            batch_size: The batch size to use with the underlying model

        Returns:
            A list of lists of annotations.
        """
        single = isinstance(images, np.ndarray) and len(images.shape) == 3
        self.model.eval()
        annotation_groups = []
        with torch.no_grad():
            for start in range(0, 1 if single else len(images), batch_size):
                current_images, current_scales = self.resize_to_model_size(
                    typing.cast(
                        typing.List[np.ndarray],
                        [images] if single else images[start : start + batch_size],
                    )
                )
                annotation_groups.extend(
                    [
                        [a.resize(1 / scale) for a in annotations]
                        for scale, annotations in zip(
                            current_scales[:, ::-1],
                            self.invert_targets(
                                self.model(
                                    self.compute_inputs(current_images),
                                ),
                                **kwargs,
                            ),
                        )
                    ]
                )
        return annotation_groups[0] if single else annotation_groups

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
                    self.detect(
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
                f.write(
                    json.dumps(
                        {
                            **{0: "__background__"},
                            **{
                                str(idx + 1): label.name
                                for idx, label in enumerate(self.annotation_config)
                            },
                        }
                    )
                )
            with open(model_file, "w", encoding="utf8") as f:
                f.write(self.serve_module_string())
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

    def load_weights(self, filepath: str):
        """Load weights from disk."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def save_weights(self, filepath: str):
        """Save weights to disk."""
        torch.save(self.model.state_dict(), filepath)
