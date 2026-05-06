#!/usr/bin/env python3

import argparse
import os
import shutil
import hashlib
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm

'''
Adapted from:
https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py
'''



def _download(url, fname, chunk_size=1024):
    '''https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51'''
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    Args:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
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


def _hash_file(fpath, algorithm='md5', chunk_size=131071):
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
    if (algorithm == 'sha256') or (algorithm == 'auto'):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='md5', chunk_size=131071):
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
    if ((algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    return str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash)


def get_file(origin=None,
             fname=None,
             file_hash=None,
             datadir='datasets',
             hash_algorithm='md5',
             extract=False,
             force_download=False,
             archive_format='auto'):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    Args:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If `None`, the
            name of the file at `origin` will be used.
        origin: Original URL of the file.
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are `'md5'`, `'sha256'`, and `'auto'`.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default `'auto'` corresponds to `['tar', 'zip']`.
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the default directory `datasets/`.
    Returns:
        Path to the downloaded file
    """
    if origin is None:
        raise ValueError('Please specify the "origin" argument (URL of the file '
                         'to download).')

    os.makedirs(datadir, exist_ok=True)

    if not fname:
        fname = os.path.basename(urllib.parse.urlsplit(origin).path)
        if not fname:
            raise ValueError(
                f"Can't parse the file name from the origin provided: '{origin}'."
                "Please specify the `fname` as the input param.")

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath) and not force_download:
        # File found; verify integrity if a hash was provided.
        print(f'A local file already found at {fpath}, checking hash...')
        if file_hash is not None:
            if validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('Local file hash matches, no need to download.')
            else:
                print(
                    'A local file was found, but it seems to be '
                    f'incomplete or outdated because the {hash_algorithm} '
                    f'file hash does not match the original value of {file_hash} '
                    'so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print(f'Downloading data from {origin} to {fpath}')

        error_msg = 'URL fetch failure on {}: {}'
        try:
            try:
                _download(origin, fpath)
            except requests.exceptions.RequestException as e:
                raise Exception(error_msg.format(origin, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise RuntimeError(f'Checksum does not match for file {fpath}')

    if extract:
        extract_archive(fpath, datadir, archive_format)

    return fpath, download


datasets = {
    'JetClass': {
        'Pythia/train_100M': [
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar', 'de4fd2dca2e68ab3c85d5cfd3bcc65c3'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part1.tar', '9722a359c5ef697bea0fbf79bf50f003'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part2.tar', '1e9f66cd1f915f9d10e90ae1d7761720'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part3.tar', '47348fc8985319fa4806da87500482fa'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part4.tar', '6b0ce16bd93b442a8d51914466990279'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part5.tar', '416e347512e716de51d392bee327b8e9'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part6.tar', 'e9b9c1557b1b39bf0a16e4ab631ae451'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part7.tar', '5bfc6cb285ccb7680cefa9ac82ad1a2e'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part8.tar', '540c1a0d66dfad78d2b363c5740ccf86'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part9.tar', '668f40b3275167ff7104c48317c0ae2a'),
        ],
        'Pythia/': [
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar', '7235ccb577ed85023ea3ab4d5e6160cf'),
            ('https://zenodo.org/record/6619768/files/JetClass_Pythia_test_20M.tar', '64e5156d26d101adeb43b8388207d767'),
        ],
    },
    'TopLandscape': {
        # converted from https://zenodo.org/record/2603256
        '../': [
            ('https://hqu.web.cern.ch/datasets/TopLandscape/TopLandscape.tar', '4fca2e47afbf321b0f201da6b804c404'),
        ],
    },
    'QuarkGluon': {
        # converted from https://zenodo.org/record/3164691
        '../': [
            ('https://hqu.web.cern.ch/datasets/QuarkGluon/QuarkGluon.tar', 'd8dd7f71a7aaaf9f1d2ee3cddef998f9'),
        ],
    },
}


def download_dataset(dataset, basedir, envfile, force_download):
    info = datasets[dataset]
    datadir = os.path.join(basedir, dataset)
    if force_download:
        if os.path.exists(datadir):
            print(f'Removing existing dir {datadir}')
            shutil.rmtree(datadir)
    for subdir, flist in info.items():
        for url, md5 in flist:
            fpath, download = get_file(url, datadir=datadir, file_hash=md5, force_download=force_download)
            if download:
                extract_archive(fpath, path=os.path.join(datadir, subdir))

    datapath = f'DATADIR_{dataset}={datadir}'
    with open(envfile) as f:
        lines = f.readlines()
    with open(envfile, 'w') as f:
        for l in lines:
            if f'DATADIR_{dataset}' in l:
                l = f'export {datapath}\n'
            f.write(l)
    print(f'Updated dataset path in {envfile} to "{datapath}".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=datasets.keys(), help='datasets to download')
    parser.add_argument('-d', '--basedir', default='datasets', help='base directory for the datasets')
    parser.add_argument('-e', '--envfile', default='env.sh', help='env file with the dataset paths')
    parser.add_argument('-f', '--force', action='store_true', help='force to re-download dataset')
    args = parser.parse_args()

    download_dataset(args.dataset, args.basedir, args.envfile, args.force)