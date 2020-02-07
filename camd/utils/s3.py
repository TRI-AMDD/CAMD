"""
Module with methods to access S3.

"""
import os

import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from camd import CAMD_CACHE, tqdm


def s3_connection_broken(bucket, prefix):
    """
    Tests if s3 connection can be established.

    Args:
        bucket: str
            name of S3 bucket
        prefix: str
            full S3 prefix of object/folder

    Returns:
        True if connection is broken
        False if connection is successfully established

    """
    try:
        s3 = boto3.client('s3')
        s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return False
    # Could add client error here, but I'm reluctant to because
    # it could lead to soft errors if we don't have permissions
    # properly set up - jhmontoya
    except NoCredentialsError as e:
        return True


def read_s3_object_to_dataframe(bucket, prefix, header=0, sep=',', encoding='utf8'):
    """
    Reads a s3 object and creates a pandas data frame object under the
    assumption that the contained data is comma separated.

    Args:
        bucket: str
            name of the S3 bucket
        prefix: str
            full prefix of the S3 object
        encoding: str
            object encoding of object (default: utf8)

    Returns: pandas.DataFrame
        data frame with data in S3 object

    """
    f = boto3.resource('s3').Object(bucket, prefix).get()['Body']
    df = pd.read_csv(f, header=header, sep=sep, encoding=encoding)
    return df


def read_s3_object_body(bucket, prefix, amt=None):
    """
    Reads a s3 object and creates a pandas data frame object under the
    assumption that the contained data is comma separated.

    Args:
        bucket: str
            name of the S3 bucket
        prefix: str
            full prefix of the S3 object
        amt: int
            bytes to read from s3 file. default=None (read complete file)

    Returns: pandas.DataFrame
        data frame with data in S3 object

    """
    f = boto3.resource('s3').Object(bucket, prefix).get()['Body']
    return f.read(amt)


def hook(t):
    """tqdm hook for processing s3 downloads"""
    def inner(bytes_amount):
        t.update(bytes_amount)
    return inner


def cache_s3_objs(obj_names, bucket_name='matr.io',
                  filter_existing_files=True):
    """
    Quick function to download relevant s3 files to cache

    Args:
        obj_names ([str]): list of object names
        bucket_name (str): name of s3 bucket
        filter_existing_files (bool): whether or not to filter existing files

    Returns:
        None
    """

    # make cache dir
    if not os.path.isdir(CAMD_CACHE):
        os.mkdir(CAMD_CACHE)

    # Initialize s3 resource
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    # Filter out existing files if desired
    if filter_existing_files:
        obj_names = [obj_name for obj_name in obj_names
                     if not os.path.isfile(os.path.join(CAMD_CACHE, obj_name))]

    for s3_key in obj_names:
        path, filename = os.path.split(s3_key)
        if not os.path.isdir(os.path.join(CAMD_CACHE, path)):
            os.makedirs(os.path.join(CAMD_CACHE, path))
        file_object = s3.Object(bucket_name, s3_key)
        file_size = file_object.content_length
        with tqdm(total=file_size, unit_scale=True, desc=s3_key) as t:
            bucket.download_file(
                s3_key, os.path.join(CAMD_CACHE, s3_key), Callback=hook(t))


def iterate_bucket_items(bucket):
    """
    Generator that iterates over all objects in a given s3 bucket

    :param bucket: name of s3 bucket
    :return: dict of metadata for an object
    """

    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket)

    for page in page_iterator:
        if page['KeyCount'] > 0:
            for item in page['Contents']:
                yield item


def s3_sync(s3_bucket, s3_prefix, sync_path="."):
    """
    Syncs a given path to an s3 prefix

    Args:
        s3_bucket (str): bucket name
        s3_prefix (str): s3 prefix to sync to
        sync_path (str, Path): path to sync to bucket:prefix

    Returns:
        (None)

    """
    # Get bucket
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(s3_bucket)

    # Walk paths and subdirectories, uploading files
    for path, subdirs, files in os.walk(sync_path):
        # Get relative path prefix
        relpath = os.path.relpath(path, sync_path)
        if not relpath.startswith('.'):
            prefix = os.path.join(s3_prefix, relpath)
        else:
            prefix = s3_prefix

        for file in files:
            file_key = os.path.join(prefix, file)
            bucket.upload_file(os.path.join(path, file), file_key)


# List of objects to sync upon running of this script
MATRIO_S3_OBJS = [
    "camd/shared-data/oqmd_voro_v3.csv",
    "camd/shared-data/oqmd1.2_icsd_featurized_clean_v2.pickle",
    "camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db"
]

if __name__ == "__main__":
    cache_s3_objs(MATRIO_S3_OBJS, bucket_name="matr.io",
                  filter_existing_files=True)
