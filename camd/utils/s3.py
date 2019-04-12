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