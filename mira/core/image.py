from typing import Union, Tuple
from os import path
import logging
import urllib
import io

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import validators
import cv2

log = logging.getLogger(__name__)


class Image(np.ndarray):
    """Provides convenience functions on top of `np.ndarray`
    for image handling.
    """
    def __init__(self, *args, **kwargs):
        return super(Image, self).__init__(*args, **kwargs)

    @staticmethod
    def read(filepath_or_buffer: Union[str, io.BytesIO]):
        """Read a file into an image object

        Args:
            filepath_or_buffer: The path to the file or any object
                with a `read` method (such as `io.BytesIO`)
        """
        if hasattr(filepath_or_buffer, 'read'):
            image = np.asarray(
                bytearray(filepath_or_buffer.read()),
                dtype=np.uint8
            )
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        elif (
            type(filepath_or_buffer) == str and
            validators.url(filepath_or_buffer)
        ):
            return Image.read(urllib.request.urlopen(filepath_or_buffer))
        else:
            assert path.isfile(filepath_or_buffer), \
                'Could not find image at path: ' + filepath_or_buffer
            image = cv2.imread(filepath_or_buffer)
        return image.view(Image).rbswap()

    @staticmethod
    def new(
        width: int,
        height: int,
        channels: int,
        cval=255
    ) -> 'Image':
        """Obtain a new blank image with given dimensions.

        Args:
            width: The width of the blank image
            height: The height of the blank image
            channels: The number of channels. If 0, the image
                will only have two dimensions (y and x).
            cval: The value to set all pixels to (does not apply to the
                alpha channel)

        Returns:
            The blank image
        """
        if channels == 0:
            image = np.zeros((height, width)) + 255
        else:
            image = np.zeros((height, width, channels)) + cval
        return np.uint8(image).view(Image)

    @property
    def width(self):
        """The width of the image"""
        return self.shape[1]

    @property
    def height(self):
        """The height of the image"""
        return self.shape[0]

    @property
    def channels(self):
        """The number of channels in the image"""
        return 0 if len(self.shape) == 2 else self.shape[2]

    def rbswap(self):
        """Swap the red and blue channels for reading and writing
        images."""
        if self.channels != 3 and self.channels != 4:
            log.info('Not swapping red and blue due to number of channels.')
            return self
        return cv2.cvtColor(self, cv2.COLOR_RGB2BGR).view(Image)

    def color(self, channels=3):
        """Convert to color image if it is not already."""
        if len(self.shape) == 3 and self.shape[2] == 1:
            return self.repeat(channels, axis=2)
        elif len(self.shape) == 2:
            image = self[:, :, np.newaxis]
            return image.repeat(channels, axis=2)
        else:
            return self

    def scaled(self, minimum: float=-1, maximum: float=1):
        """Obtain a scaled version of the image with values between
        minimum and maximum.

        Args:
            minimum: The minimum value
            maximum: The maximum value

        Returns:
            An array of same shape as image but of dtype `np.float32` with
            values scaled appropriately.
        """
        assert maximum > minimum
        x = np.float32(self)
        x /= 255
        x *= (maximum - minimum)
        x += minimum
        return x

    def fit(
        self,
        width: int=None,
        height: int=None,
        cval: int=255
    ) -> Tuple['Image', float]:
        """Obtain a new image, fit to the specified size.

        Args:
            width: The new width
            height: The new height
            cval: The constant value to use to fill the remaining areas of
                the image

        Returns:
            The new image and the scaling that was applied.
        """
        if width == self.width and height == self.height:
            return self, 1
        scale = min(width / self.width, height / self.height)
        fitted = Image.new(
            width=width,
            height=height,
            channels=self.channels,
            cval=cval
        ).view(Image)
        image = self.resize(scale=scale)
        fitted[:image.height, :image.width] = image[:width, :height]
        return fitted, scale

    def resize(self, scale: float, interpolation=cv2.INTER_NEAREST):
        """Obtain resized version of image with a given scale

        Args:
            scale: The scale by which to resize the image
            interpolation: The interpolation method to use

        Returns:
            The scaled image
        """
        width = int(np.ceil(scale*self.width))
        height = int(np.ceil(scale*self.height))
        resized = cv2.resize(
            self,
            dsize=(width, height),
            interpolation=interpolation
        ).view(Image)
        if len(resized.shape) == 2 and len(self.shape) == 3:
            # This was a grayscale image and we need it to be returned
            # as such.
            resized = resized[:, :, np.newaxis]
        return resized

    def buffer(self, extension='.jpg') -> io.BytesIO:
        """Convert the image to a BytesIO object with encoding
        for the given extension.

        Args:
            extension: The extension of the target file format,
                such as '.jpg'

        Returns:
            The encoded file
        """
        image = cv2.imencode(extension, self.rbswap())[1].tobytes()
        return io.BytesIO(image)

    def save(
        self,
        filepath_or_buffer: Union[str, io.BytesIO],
        extension='.jpg'
    ):
        """Save the image

        Args:
            filepath_or_buffer: The file or buffer to
                which to save the image. If buffer,
                format must be provided
            esxtension: The extension for the format to use
                if writing to buffer
        """
        image = self.rbswap()
        if hasattr(filepath_or_buffer, 'write'):
            data = cv2.imencode(extension, image)[1].tobytes()
            filepath_or_buffer.write(data)
        else:
            cv2.imwrite(filepath_or_buffer, img=image)

    def show(self, ax: mpl.axes.Axes=None) -> mpl.axes.Axes:
        """Show an image

        Args:
            ax: Axis on which to show the image

        Returns:
            An axes object
        """
        if ax is None:
            ax = plt
        if len(self.shape) == 3 and self.shape[2] >= 3:
            return ax.imshow(self)
        elif len(self.shape) == 3 and self.shape[2] == 1:
            return ax.imshow(self[:, :, 0])
        elif len(self.shape) == 2:
            return ax.imshow(self)
        else:
            raise ValueError('Incorrect dimensions for image data.')
        return ax.imshow(self)