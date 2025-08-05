# pylint: disable=too-many-public-methods
import os
import abc
import json
import types
import typing
import logging
import tempfile
from importlib import resources

import tqdm
import torch
import numpy as np

import typing_extensions as tx

from .. import metrics as mm
from .. import core as mc

LOGGER = logging.getLogger(__name__)


class Detector(mc.torchtools.BaseModel):
    """Abstract base class for a detector."""

    @abc.abstractmethod
    def serve_module_string(self) -> str:
        """Return the module string used as part of TorchServe."""

    @abc.abstractmethod
    def compute_anchor_boxes(self, width: int, height: int) -> np.ndarray:
        """Return the list of anchor boxes in xyxy format. You can convert these
        to dimensions using something like:

        detector.compute_anchor_boxes(iwidth, iheight)[:, [0, 2, 1, 3]].reshape((-1, 2, 2))
        """

    def detect(
        self,
        items: mc.torchtools.BatchInferenceItem,
        batch_size: int = 32,
        progress=False,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[mc.Annotation]],
        typing.List[mc.Annotation],
        mc.SceneCollection,
        mc.Scene,
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
        single, annotation_groups = self.batch_inference(
            items=items,
            batch_size=batch_size,
            progress=progress,
            process=lambda batch: [
                [a.resize(1 / scale) for a in itarget.annotations]
                for scale, itarget in zip(
                    batch.scales[:, ::-1],
                    self.invert_targets(
                        self.model(
                            self.compute_inputs(batch.images),
                        ),
                        **kwargs,
                    ),
                )
            ],
        )
        if isinstance(items, mc.SceneCollection):
            return items.assign(
                scenes=[
                    s.assign(annotations=g) for s, g in zip(items, annotation_groups)
                ]
            )
        if isinstance(items, mc.Scene):
            return items.assign(annotations=annotation_groups[0])
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
        return mm.mAP(
            true_collection=collection,
            pred_collection=typing.cast(
                mc.SceneCollection,
                self.detect(
                    collection,
                    threshold=min_threshold,
                    batch_size=batch_size,
                ),
            ),
            iou_threshold=iou_threshold,
        )

    # pylint: disable=import-outside-toplevel
    def to_torchserve(
        self,
        model_name: str,
        directory=".",
        archive_format: tx.Literal["default", "no-archive"] = "default",
        score_threshold: float = 0.5,
        model_version="1.0",
        api_mode: tx.Literal["mira", "torchserve"] = "mira",
    ):
        """Build a TorchServe-compatible MAR file for this model."""
        try:
            import model_archiver.model_packaging as marmp
            import model_archiver.model_packaging_utils as marmpu
        except ImportError as e:
            raise ValueError(
                "You must `pip install torch-model-archiver` to use this function."
            ) from e
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
                                for idx, label in enumerate(self.categories)
                            },
                        }
                    )
                )
            with open(model_file, "w", encoding="utf8") as f:
                f.write(self.serve_module_string())
            with open(handler_file, "w", encoding="utf8") as f:
                f.write(
                    resources.files("mira")
                    .joinpath("detectors/assets/serve/object_detector.py")
                    .read_text("utf8")
                    .replace("SCORE_THRESHOLD", str(score_threshold))  # type: ignore
                    .replace("API_MODE", f"'{api_mode}'")
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

    def compute_anchor_iou(
        self, collection: mc.SceneCollection
    ) -> typing.List[np.ndarray]:
        """Compute the IoU between annotatons for a scene collection and the anchors for the detector.
        Accounts for scaling depending on this detectors resize configuration."""
        dimensions, scales, _ = mc.resizing.compute_resize_dimensions(
            np.array(
                [
                    [scene.dimensions.height, scene.dimensions.width]
                    for scene in collection
                ]
            ),
            self.resize_config,
        )
        bboxes = [
            np.array(
                [ann.resize(scale[::-1]).x1y1x2y2() for ann in scene.annotations],
                ndmin=2,
            )
            for scene, scale in zip(collection, scales)
        ]
        anchors = self.compute_anchor_boxes(
            height=dimensions[0][0], width=dimensions[0][1]
        )
        return [
            mc.utils.compute_iou(group, anchors)
            for group in tqdm.tqdm(bboxes, desc="Computing IoU matrices.")
        ]
