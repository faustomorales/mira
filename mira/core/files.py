"""Core utilities"""
import typing
import logging
import os
import hashlib
import io
import shutil
import tarfile
import zipfile
import requests

from tqdm import tqdm

log = logging.getLogger(__name__)

# Progbar, _extract_archive, get_file, _hash_file and validate_file
# adapted / copied from Keras to avoid requiring
# TensorFlow to simply use file downloads, etc.

# ProgressFileObject adapted from https://stackoverflow.com/a/3668977


def get_datadir_base():
    """Get the base directory for data."""
    cache_dir_env = os.environ.get("MIRA_CACHE")
    if cache_dir_env is not None:
        cache_dir = os.path.abspath(cache_dir_env)
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".mira")
    return os.path.expanduser(cache_dir)


class ProgressFileObject(io.FileIO):  # noqa: E302
    """A file downloader that reports progress."""

    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        self.progbar = tqdm(total=self._total_size, desc="Reading " + path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size=-1):
        if size > 0:
            self.progbar.update(size)
        return io.FileIO.read(self, size)

    def close(self):
        self.progbar.close()
        super().close()


# pylint: disable=consider-using-with
def extract_archive(file_path, path=".", extract_check_fn=None):  # noqa: E302
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    open_fn: typing.Callable[
        [typing.Any], typing.Union[tarfile.TarFile, zipfile.ZipFile]
    ]
    if tarfile.is_tarfile(file_path):
        open_fn = lambda fp: tarfile.open(
            fileobj=ProgressFileObject(fp)
        )  # noqa: E731,E501
    if zipfile.is_zipfile(file_path):
        open_fn = lambda fp: zipfile.ZipFile(
            file=ProgressFileObject(fp)
        )  # noqa: E731,E501

    if not os.path.exists(path) or (
        extract_check_fn is not None and not extract_check_fn(path)
    ):
        with open_fn(file_path) as archive:
            try:
                archive.extractall(path)
            except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                raise
            assert extract_check_fn is None or extract_check_fn(
                path
            ), "Extraction succeeded but the check failed."
    return path


def compute_sha256(fpath):
    """Compute sha256 for file."""
    hasher = hashlib.sha256()
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(65535), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file(
    origin,
    fname,
    file_hash=None,
    cache_subdir="datasets",
    extract=False,
    extract_check_fn=None,
):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If None,
            the name of the file at origin is used instead.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    # Returns
        Path to the downloaded file
    """
    datadir_base = get_datadir_base()
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)
    if not os.path.exists(fpath) or (
        file_hash is not None and compute_sha256(fpath) != file_hash
    ):
        try:
            with requests.get(origin, stream=True, timeout=10) as response:
                response.raise_for_status()
                with tqdm(
                    total=int(response.headers.get("content-length", 0)),
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading from {origin} to {fpath}.",
                ) as progbar, open(fpath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        progbar.update(len(chunk))
                        f.write(chunk)
        except Exception:  # pylint: disable=broad-except
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    if extract:
        extract_target = os.path.join(
            datadir, os.path.split(fpath)[1].split(os.extsep)[0]
        )
        return extract_archive(
            file_path=fpath,
            path=extract_target,
            extract_check_fn=extract_check_fn,
        )
    return fpath
