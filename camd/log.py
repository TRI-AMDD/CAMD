# Copyright 2019, Toyota Research Institute
"""
Logging configuration for operation of CAMD

CAMD logging can be configured by two environment variables:

CAMD_LOG_CONFIG (str): a string that defines the logging handlers
    to be added.  If the string contains "kinesis", the kinesis
    handler is added to the logger, if "stdout", the log is printed
    to stdout.

    Examples:
        "kinesis" - logs to kinesis, kinesis log name can be specified
            via CAMD_KINESIS_STREAM_NAME below
        "stdout" - logs only to stdout
        "kinesis,stdout" - logs to both kinesis and stdout

CAMD_KINESIS_STREAM_NAME (str): stream name for CAMD kinesis
    events.  Defaults to "kinesis-test" but can be configured,
    e. g. to camd-events for production runs

"""
import logging, sys, os
from autologging import TRACE
import boto3


class KinesisHandler(logging.Handler):
    """
    Simple handler for kinesis logging
    """

    def __init__(self, level, stream_name, **kwargs):
        super(KinesisHandler, self).__init__(level=level)
        self.stream_name = stream_name
        self.client = boto3.client('kinesis', **kwargs)

    def emit(self, record):
        try:
            self.client.put_record(StreamName=self.stream_name,
                                   Data=self.format(record),
                                   PartitionKey=str(hash('test'))
                                   )
        except:
            self.handleError(record)


CAMD_LOG_FORMAT = "%(levelname)s:%(name)s:%(funcName)s:%(message)s"
CAMD_LOGGER = logging.getLogger('root')
CAMD_LOGGER.setLevel(TRACE)
CAMD_LOG_FORMATTER = logging.Formatter(CAMD_LOG_FORMAT)

STREAM_HANDLER = logging.StreamHandler(sys.stdout)
STREAM_HANDLER.setLevel(TRACE)
STREAM_HANDLER.setFormatter(CAMD_LOG_FORMATTER)

KINESIS_STREAM_NAME = os.environ.get("CAMD_KINESIS_STREAM_NAME", "kinesis-test")
KINESIS_HANDLER = KinesisHandler(TRACE, "kinesis-test",
                                 region_name="us-west-2")
KINESIS_HANDLER.setFormatter(CAMD_LOG_FORMATTER)

CAMD_LOG_CONFIG = os.environ.get("CAMD_LOG_CONFIG", "")

if "kinesis" in CAMD_LOG_CONFIG:
    CAMD_LOGGER.addHandler(KINESIS_HANDLER)

if "stdout" in CAMD_LOG_CONFIG:
    CAMD_LOGGER.addHandler(STREAM_HANDLER)
