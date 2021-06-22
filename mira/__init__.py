import warnings

warnings.filterwarnings(
    action="ignore",
    message=r".*The output shape of `ResNet50\(include_top=False\)` has been changed since Keras 2\.2\.0\..*",
    category=UserWarning,
)
__version__ = "0.0.0"
