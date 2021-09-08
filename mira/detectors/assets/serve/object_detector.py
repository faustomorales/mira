# type: ignore

from ts.torch_handler.object_detector import ObjectDetector


class ThresholdConfigurableObjectDetector(ObjectDetector):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    def __init__(self):
        super().__init__()
        self.threshold = SCORE_THRESHOLD
