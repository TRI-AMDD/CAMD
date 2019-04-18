"""
Module with methods to access S3.

"""
import os

import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from camd import S3_CACHE, tqdm


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


def sync_s3_objs():
    """Quick function to download relevant s3 files to cache"""

    # make cache dir
    if not os.path.isdir(S3_CACHE):
        os.mkdir(S3_CACHE)

    # Initialize s3 resource
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("ml-dash-datastore")

    # Put more s3 objects here if desired
    s3_keys = ["oqmd_voro_March25_v2.csv"]

    s3_keys_to_download = set(s3_keys) - set(os.listdir(S3_CACHE))

    for s3_key in s3_keys_to_download:
        filename = os.path.split(s3_key)[-1]
        file_object = s3.Object("ml-dash-datastore", s3_key)
        file_size = file_object.content_length
        with tqdm(total=file_size, unit_scale=True, desc=filename) as t:
            bucket.download_file(
                s3_key, os.path.join(S3_CACHE, filename), Callback=hook(t))