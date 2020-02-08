"""
Module with methods to access S3.

"""

import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd


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


