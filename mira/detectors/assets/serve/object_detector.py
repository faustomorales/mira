# type: ignore
import os
import time
from ts.torch_handler.object_detector import ObjectDetector


class ThresholdConfigurableObjectDetector(ObjectDetector):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    def handle(self, data, context):
        start_time = time.time()
        self.context = context
        if os.environ.get("ENABLE_TORCH_PROFILER", None):
            raise RuntimeError(
                "Profiler is enabled but this detector does not support it."
            )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)
                output = self.inference(data_preprocess)
                output = self.postprocess(
                    output,
                    thresholds=[
                        float(
                            context.get_request_header(idx, "X-DETECTION-THRESHOLD")
                            or SCORE_THRESHOLD
                        )
                        for idx in range(len(data))
                    ],
                )
        self.context.metrics.add_time(
            "HandlerTime", round((time.time() - start_time) * 1000, 2), None, "ms"
        )
        return output

    def postprocess(self, data, thresholds):
        return [
            [
                {
                    "score": score,
                    "label": self.mapping[str(label)],
                    **(
                        dict(zip(["x1", "y1", "x2", "y2"], box))
                        if "boxes" in row
                        else dict(polygon=box)
                    ),
                }
                for score, box, label in zip(
                    *[
                        l if isinstance(l, list) else l.tolist()
                        for l in [
                            row["scores"],
                            row["boxes" if "boxes" in row else "polygons"],
                            row["labels"],
                        ]
                    ]
                )
                if score > threshold
            ]
            for row, threshold in zip(data, thresholds)
        ]
