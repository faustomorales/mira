# pylint: disable=broad-except
"""
# Taken from https://github.com/scardine/image_size

get_image_size.py
====================

    :Name:        get_image_size
    :Purpose:     extract image dimensions given a file path

    :Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)

    :Created:     26/09/2013
    :Copyright:   (c) Paulo Scardine 2013
    :Licence:     MIT

"""

import os
import io
import typing
import struct
import warnings

FILE_UNKNOWN = "Sorry, don't know how to get size for this file."


class UnknownImageFormat(Exception):
    """An exception for when we don't recognize the image format."""


ImageMeta = typing.NamedTuple(
    "ImageMeta", [("width", int), ("height", int), ("file_size", int), ("type", str)]
)


def get_image_size_from_bytesio(finput, size) -> typing.Tuple[int, int]:
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        finput (io.IOBase): io object support read & seek
        size (int): size of buffer in byte
    """
    img = get_image_metadata_from_bytesio(finput, size)
    return (img.width, img.height)


def get_image_metadata(file_path) -> ImageMeta:
    """
    Return an `Image` object for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        file_path (str): path to an image file

    Returns:
        Image: (path, type, file_size, width, height)
    """
    size = os.path.getsize(file_path)

    # be explicit with open arguments - we need binary mode
    with io.open(file_path, "rb") as finput:
        return get_image_metadata_from_bytesio(finput, size)


def get_image_metadata_from_bytesio(finput, size) -> ImageMeta:
    """
    Return an `Image` object for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        finput (io.IOBase): io object support read & seek
        size (int): size of buffer in byte
        file_path (str): path to an image file

    Returns:
        Image: (path, type, file_size, width, height)
    """
    height = -1
    width = -1
    data = finput.read(26)
    msg = " raised while trying to decode as JPEG."

    if (size >= 10) and data[:6] in (b"GIF87a", b"GIF89a"):
        # GIFs
        imgtype = "GIF"
        w, h = struct.unpack("<HH", data[6:10])
        width = int(w)
        height = int(h)
    elif (
        (size >= 24)
        and data.startswith(b"\211PNG\r\n\032\n")
        and (data[12:16] == b"IHDR")
    ):
        # PNGs
        imgtype = "PNG"
        w, h = struct.unpack(">LL", data[16:24])
        width = int(w)
        height = int(h)
    elif (size >= 16) and data.startswith(b"\211PNG\r\n\032\n"):
        # older PNGs
        imgtype = "PNG"
        w, h = struct.unpack(">LL", data[8:16])
        width = int(w)
        height = int(h)
    elif (size >= 2) and data.startswith(b"\377\330"):
        # JPEG
        imgtype = "JPEG"
        finput.seek(0)
        finput.read(2)
        b = finput.read(1)
        try:
            while b and ord(b) != 0xDA:
                while ord(b) != 0xFF:
                    b = finput.read(1)
                while ord(b) == 0xFF:
                    b = finput.read(1)
                if ord(b) >= 0xC0 and ord(b) <= 0xC3:
                    finput.read(3)
                    h, w = struct.unpack(">HH", finput.read(4))
                    break
                finput.read(int(struct.unpack(">H", finput.read(2))[0]) - 2)
                b = finput.read(1)
            width = int(w)
            height = int(h)
        except struct.error as exc:
            raise UnknownImageFormat("StructError" + msg) from exc
        except ValueError as exc:
            raise UnknownImageFormat("ValueError" + msg) from exc
        except Exception as exc:
            raise UnknownImageFormat(exc.__class__.__name__ + msg) from exc
    elif (size >= 26) and data.startswith(b"BM"):
        # BMP
        imgtype = "BMP"
        headersize = struct.unpack("<I", data[14:18])[0]
        if headersize == 12:
            w, h = struct.unpack("<HH", data[18:22])
            width = int(w)
            height = int(h)
        elif headersize >= 40:
            w, h = struct.unpack("<ii", data[18:26])
            width = int(w)
            # as h is negative when stored upside down
            height = abs(int(h))
        else:
            raise UnknownImageFormat("Unkown DIB header size:" + str(headersize))
    elif (size >= 8) and data[:4] in (b"II\052\000", b"MM\000\052"):
        # Standard TIFF, big- or little-endian
        # BigTIFF and other different but TIFF-like formats are not
        # supported currently
        imgtype = "TIFF"
        byteOrder = data[:2]
        boChar = ">" if byteOrder == "MM" else "<"
        # maps TIFF type id to size (in bytes)
        # and python format char for struct
        tiffTypes = {
            1: (1, boChar + "B"),  # BYTE
            2: (1, boChar + "c"),  # ASCII
            3: (2, boChar + "H"),  # SHORT
            4: (4, boChar + "L"),  # LONG
            5: (8, boChar + "LL"),  # RATIONAL
            6: (1, boChar + "b"),  # SBYTE
            7: (1, boChar + "c"),  # UNDEFINED
            8: (2, boChar + "h"),  # SSHORT
            9: (4, boChar + "l"),  # SLONG
            10: (8, boChar + "ll"),  # SRATIONAL
            11: (4, boChar + "f"),  # FLOAT
            12: (8, boChar + "d"),  # DOUBLE
        }
        ifdOffset = struct.unpack(boChar + "L", data[4:8])[0]
        try:
            countSize = 2
            finput.seek(ifdOffset)
            ec = finput.read(countSize)
            ifdEntryCount = struct.unpack(boChar + "H", ec)[0]
            # 2 bytes: TagId + 2 bytes: type + 4 bytes: count of values + 4
            # bytes: value offset
            ifdEntrySize = 12
            for i in range(ifdEntryCount):
                entryOffset = ifdOffset + countSize + i * ifdEntrySize
                finput.seek(entryOffset)
                tag = finput.read(2)
                tag = struct.unpack(boChar + "H", tag)[0]
                if tag in (256, 257):
                    # if type indicates that value fits into 4 bytes, value
                    # offset is not an offset but value itself
                    ftype = finput.read(2)
                    ftype = struct.unpack(boChar + "H", ftype)[0]
                    if ftype not in tiffTypes:
                        raise UnknownImageFormat("Unkown TIFF field type:" + str(ftype))
                    typeSize = tiffTypes[ftype][0]
                    typeChar = tiffTypes[ftype][1]
                    finput.seek(entryOffset + 8)
                    value = finput.read(typeSize)
                    value = int(struct.unpack(typeChar, value)[0])
                    if tag == 256:
                        width = value
                    else:
                        height = value
                if width > -1 and height > -1:
                    break
        except Exception as exc:
            raise UnknownImageFormat(str(exc)) from exc
    elif size >= 2:
        # see http://en.wikipedia.org/wiki/ICO_(file_format)
        imgtype = "ICO"
        finput.seek(0)
        reserved = finput.read(2)
        if 0 != struct.unpack("<H", reserved)[0]:
            raise UnknownImageFormat(FILE_UNKNOWN)
        fformat = finput.read(2)
        assert 1 == struct.unpack("<H", fformat)[0]
        num = finput.read(2)
        num = struct.unpack("<H", num)[0]
        if num > 1:
            warnings.warn("ICO File contains more than one image")
        # http://msdn.microsoft.com/en-us/library/ms997538.aspx
        w = finput.read(1)
        h = finput.read(1)
        width = ord(w)
        height = ord(h)
    else:
        raise UnknownImageFormat(FILE_UNKNOWN)

    return ImageMeta(type=imgtype, file_size=size, width=width, height=height)
