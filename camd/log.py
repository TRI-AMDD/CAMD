# Copyright 2019, Toyota Research Institute
import logging, sys
from autologging import traced, TRACE
from aws_logging_handlers.Kinesis import KinesisHandler

# logging.basicConfig(
#     level=TRACE, stream=sys.stdout,
#     format="%(levelname)s:%(name)s:%(funcName)s:%(message)s"
# )

logger = logging.getLogger("root")
formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")
logger.setLevel(TRACE)
kinesis_handler = KinesisHandler("kinesis-test", "us-west-2", )
logger.addHandler(kinesis_handler)

@traced
def print_and_return(something):
    print(something)
    return something


if __name__ == "__main__":
    print_and_return(something="Hello world!")
