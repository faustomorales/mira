"""Core utilities"""
import typing
import logging
import os
import hashlib
import io
import shutil
import tarfile
import zipfile
import urllib.parse
import urllib.request
import urllib.error

import six
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
def _extract_archive(  # noqa: E302
    file_path, path=".", archive_format="auto", extract_check_fn=None
):
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
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        open_fn: typing.Callable[
            [typing.Any], typing.Union[tarfile.TarFile, zipfile.ZipFile]
        ]
        is_match_fn: typing.Callable[[typing.Any], bool]
        if archive_type == "tar":
            # Implement progress bar for tar files,
            # per https://stackoverflow.com/questions/3667865/python-tarfile-progress-output  # pylint: disable=line-too-long
            # open_fn = tarfile.open
            open_fn = lambda fp: tarfile.open(
                fileobj=ProgressFileObject(fp)
            )  # noqa: E731,E501
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            # open_fn = zipfile.ZipFile
            open_fn = lambda fp: zipfile.ZipFile(
                file=ProgressFileObject(fp)
            )  # noqa: E731,E501
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            if os.path.exists(path) and extract_check_fn is not None:
                log.info(
                    "Found target folder. Checking for " "existing complete extraction."
                )
                if extract_check_fn(path):
                    log.info("Found complete extraction.")
                    return True
                log.info("Extraction is not complete. Re-extracting.")
            log.info("Extracting %s to %s", file_path, path)
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
            return True
    return False


def get_file(
    origin,
    fname=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
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
    if fname is None:
        fname = os.path.basename(urllib.parse.urlparse(origin).path)
    datadir_base = get_datadir_base()
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                log.warning(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated because the %s"
                    " file hash does not match the original value of %s"
                    " so we will re-download the data.",
                    hash_algorithm,
                    file_hash,
                )
                download = True
    else:
        download = True

    if download:
        error_msg = "URL fetch failure on {} : {}"
        try:
            try:
                urllib.request.urlretrieve(origin, fpath)
            except urllib.error.HTTPError as e:  # pylint: disable=invalid-name
                raise Exception(error_msg.format(origin, e.code)) from e
            except urllib.error.URLError as e:  # pylint: disable=invalid-name
                raise Exception(error_msg.format(origin, e.errno)) from e
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    if extract:
        extract_target = os.path.join(
            datadir, os.path.split(fpath)[1].split(os.extsep)[0]
        )
        extract_result = _extract_archive(
            file_path=fpath,
            path=extract_target,
            archive_format=archive_format,
            extract_check_fn=extract_check_fn,
        )
        if extract_result:
            fpath = extract_target

    return fpath


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    # Example

    ```python
        >>> from keras.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        The file hash
    """
    if algorithm in ("sha256", "auto"):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        Whether the file is valid
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"
    log.info("Checking hash for %s.", os.path.split(fpath)[1])
    return str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash)
