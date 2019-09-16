#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Module and script for running the CAMD campaign worker

Usage:
    camd_worker COMMAND --campaign CAMPAIGN

Options:
    --campaign       campaign name [default: proto-dft]
    -h --help        Show this screen
    --version        Show version

"""
import time
import boto3
import itertools
import os
import numpy as np

from pathlib import Path
from docopt import docopt
from monty.tempfile import ScratchDir
from camd import CAMD_S3_BUCKET, CAMD_STOP_FILE
from camd.log import camd_traced
from camd.campaigns.structure_discovery import run_proto_dft_campaign, run_atf_campaign


# TODO: set up test bucket, instead of using complicated pathing
@camd_traced
class Worker(object):
    def __init__(self, campaign="proto-dft"):
        self.campaign = campaign

    def start(self, num_loops=np.inf, sleep_time=60):
        """
        Starts the worker, which monitors s3 for new submissions
        and starts runs accordingly.

        Args:
            num_loops (int): number of campaigns to run before
                stopping
            sleep_time (float): time to sleep between iterations
                in which active campaigns are found

        Returns:
            (int) number of loops executed, including "sleep" loops,
                where no campaign was executed

        """
        for loop_num in itertools.count(1):
            if loop_num > num_loops or self.check_stop_file():
                return loop_num - 1
            latest_chemsys = self.get_latest_chemsys()
            if latest_chemsys:
                with ScratchDir('.') as sd:
                    print("Running {} in {}".format(latest_chemsys, sd))
                    self.run_campaign(latest_chemsys)
            else:
                print("No new campaigns submitted, sleeping for 60 seconds")
                time.sleep(sleep_time)

    def run_campaign(self, chemsys):
        """
        Runs the campaign for a given chemsys

        Args:
            chemsys ([str]):

        Returns:

        """
        if self.campaign == "proto-dft":
            run_proto_dft_campaign(chemsys)
        elif self.campaign == "oqmd-atf":
            # This is more or less just a test
            run_atf_campaign(chemsys)
        else:
            raise ValueError("Campaign {} is not valid".format(self.campaign))

    def get_latest_chemsys(self):
        bucket = boto3.resource("s3").Bucket(CAMD_S3_BUCKET)

        # Get submissions
        submission_prefix = '/'.join([self.campaign, "submit"])
        # TODO: fix 1000 return value limit - MAT-838
        submit_objects = bucket.objects.filter(Prefix=submission_prefix)
        submission_times = {obj.key.split('/')[-2]: obj.get()['LastModified']
                            for obj in submit_objects
                            if obj.key != "{}/submit/".format(self.campaign)}

        # Get started jobs
        started = get_common_prefixes(CAMD_S3_BUCKET, "/".join([self.campaign, "runs"]))

        # Filter started jobs and then get latest unstarted
        unstarted = list(set(submission_times.keys()) - set(started))
        latest_unstarted = sorted(unstarted, key=lambda x: submission_times[x])

        return latest_unstarted[-1] if latest_unstarted else None

    @staticmethod
    def write_stop_file():
        """
        Touches stop file to signal drawdown to shared workers

        Returns:
            (None)

        """
        Path(CAMD_STOP_FILE).touch()

    @staticmethod
    def remove_stop_file():
        """
        Removes the stop file from its specified location
        via CAMD_STOP_FILE env variable

        Returns:
            (None)

        """
        if os.path.isfile(CAMD_STOP_FILE):
            os.remove(CAMD_STOP_FILE)

    @staticmethod
    def check_stop_file():
        """
        Returns bool corresponding to whether the CAMD stop
        file exists

        Returns:
            (bool) whether CAMD stop file exists

        """
        return os.path.isfile(CAMD_STOP_FILE)


# TODO: move this to TRI-utils
def get_common_prefixes(bucket, prefix):
    """
    Helper function to get common "subfolders" of folders
    in S3

    Args:
        bucket (str): bucket name
        prefix (str): prefix for which to list common prefixes

    Returns:

    """
    if not prefix.endswith('/'):
        prefix += "/"
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects')
    result = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix)
    return [common_prefix['Prefix'].split('/')[-2]
            for common_prefix in result.search("CommonPrefixes")
            if common_prefix]


def main():
    args = docopt(__doc__)
    worker = Worker(args["--campaign"])
    if args['COMMAND'] == "start":
        worker.remove_stop_file()
        worker.start()
    elif args['COMMAND'] == "stop":
        worker.write_stopfile()
    else:
        raise ValueError("Invalid command {}.  Worker command must be 'start' or 'stop'")


if __name__ == "__main__":
    main()
