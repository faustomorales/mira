import abc
import typing
import logging

import typing_extensions as tx

from .. import core as mc

TrainSplitState = tx.TypedDict(
    "TrainSplitState", {"true": typing.List[int], "pred": typing.List[int]}
)
TrainState = tx.TypedDict(
    "TrainState", {"train": TrainSplitState, "val": TrainSplitState}
)

LOGGER = logging.getLogger(__name__)


class Classifier(mc.torchtools.BaseModel, metaclass=abc.ABCMeta):
    """Abstract base class for classifier."""

    def classify(
        self,
        items: mc.torchtools.BatchInferenceItem,
        batch_size=32,
        progress=False,
        threshold: float = 0.0,
    ) -> typing.Union[
        typing.List[mc.Label],
        typing.List[typing.List[mc.Label]],
        mc.SceneCollection,
        mc.Scene,
    ]:
        """Classify a batch of images."""
        single, predictions = self.batch_inference(
            items=items,
            batch_size=batch_size,
            progress=progress,
            process=lambda batch: [
                itarget.labels
                for itarget in self.invert_targets(
                    self.model(self.compute_inputs(batch.images)), threshold=threshold
                )
            ],
        )
        if isinstance(items, mc.SceneCollection):
            return items.assign(
                scenes=[s.assign(labels=g) for s, g in zip(items, predictions)]
            )
        if isinstance(items, mc.Scene):
            return items.assign(labels=predictions[0])
        return predictions[0] if single else predictions
