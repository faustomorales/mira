import warnings

warnings.filterwarnings(
    action="ignore",
    message=r'.*The output shape of `ResNet50\(include_top=False\)` has been changed since Keras 2\.2\.0\..*',  # noqa: E501
    category=UserWarning
)
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
