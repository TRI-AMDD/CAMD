#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Module and script for running the CAMD campaign worker

Usage:
    camd_worker COMMAND [options]

Options:
    --campaign CAMPAIGN  campaign name  [default: proto-dft-2]
    --loops NUM_LOOPS    number of loops to run
    --catch_errors       whether or not to catch errors
    -h --help            Show this screen
    --version            Show version

COMMAND may be one of:
    start - starts worker
    stop - writes stopfile such that CAMD stops
"""
import time
import boto3
import itertools
import os
import numpy as np
import traceback

from monty.serialization import dumpfn
from pathlib import Path
from docopt import docopt
from monty.tempfile import ScratchDir
from camd import CAMD_S3_BUCKET, CAMD_STOP_FILE
from camd.campaigns.structure_discovery import \
    ProtoDFTCampaign, CloudATFCampaign
from camd.campaigns.meta_agent import MetaAgentCampaign


class Worker(object):
    """
    The Worker is an object that is intended to
    persistently poll s3 for new submissions
    from which to start campaigns.  Currently
    primarily used for structure discovery,
    i.e. 'proto-dft' campaigns.
    """
    def __init__(self, campaign="proto-dft-2",
                 catch_errors=False):
        """
        Initialize a Worker

        Args:
            campaign (str): campaign name, e.g. 'proto-dft-2'

        """

        self.campaign = campaign
        self.catch_errors = catch_errors

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
            latest_submission = self.get_latest_submission()
            if latest_submission:
                with ScratchDir('.') as sd:
                    print("Running {} in {}".format(latest_submission, sd))
                    self.run_campaign(latest_submission)
            else:
                print("No new campaigns submitted, sleeping for {} seconds".format(
                    sleep_time))
                time.sleep(sleep_time)

    def run_campaign(self, *args):
        """
        Runs the campaign for a given chemsys

        Args:
            *args: args for a given campaign,
                specific to invocation method below i. e.

                proto-dft/oqmd-atf require CHEMSYS
                meta-agent requires NAME

        Returns:
            None

        """
        if self.campaign.startswith("proto-dft-high"):
            campaign = ProtoDFTCampaign.from_chemsys_high_quality(*args)
        elif self.campaign.startswith("proto-dft"):
            campaign = ProtoDFTCampaign.from_chemsys(*args)
        elif self.campaign.startswith("oqmd-atf"):
            # This is more or less just a test
            campaign = CloudATFCampaign.from_chemsys(*args)
        elif self.campaign.startswith("meta-agent"):
            # For meta-agent campaigns, submit with meta_agent/CAMPAIGN_NAME
            campaign = MetaAgentCampaign.from_reserved_name(*args)
        else:
            raise ValueError("Campaign {} is not valid".format(self.campaign))

        # Capture errors and store
        if self.catch_errors:
            try:
                campaign.autorun()
            except Exception as e:
                error_msg = {"error": "{}".format(e),
                             "traceback": traceback.format_exc()}
                campaign.logger.info("Error: {}".format(e))
                campaign.logger.info("Traceback: {}".format(traceback.format_exc()))
                dumpfn(error_msg, "error.json")
                dumpfn({"status": "error"}, "job_status.json")
                campaign.s3_sync()
        else:
            campaign.autorun()

    def get_latest_submission(self):
        """
        Gets the last submitted chemsys

        Returns:
            (str): corresponding to last submitted chemsys

        """
        bucket = boto3.resource("s3").Bucket(CAMD_S3_BUCKET)

        # Get submissions
        submission_prefix = '/'.join([self.campaign, "submit"])
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
    """
    Main function for running this as a script

    Returns:
        None
    """
    args = docopt(__doc__)
    campaign = args['--campaign']
    catch_errors = args['--catch_errors']
    worker = Worker(campaign, catch_errors=catch_errors)
    num_loops = args['--loops']
    if num_loops is not None:
        num_loops = int(num_loops)
    else:
        num_loops = np.inf
    if args['COMMAND'] == "start":
        print("Starting {} worker with {} loops".format(
            campaign, num_loops))
        worker.remove_stop_file()
        worker.start(num_loops=num_loops)
    elif args['COMMAND'] == "stop":
        worker.write_stop_file()
    else:
        raise ValueError("Invalid command {}.  Worker command must be 'start' or 'stop'")


if __name__ == "__main__":
    main()
