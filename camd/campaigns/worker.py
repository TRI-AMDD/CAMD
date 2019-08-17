#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Initial worker to run structure discovery.  WIP.
"""
import time
import boto3
import re

from monty.tempfile import ScratchDir
from camd import CAMD_S3_BUCKET
from camd.log import camd_traced
from camd.campaigns.structure_discovery import run_proto_dft_campaign, run_atf_campaign
import itertools


# TODO: set up test bucket, instead of using complicated pathing
@camd_traced
class Worker(object):
    def __init__(self, campaign="proto-dft"):
        self.campaign = campaign

    def start(self, num_loops=None):
        """
        Starts the worker, which monitors s3 for new submissions
        and starts runs accordingly.

        Args:
            num_loops (int): number of campaigns to run before
                stopping

        Returns:

        """
        for campaign_num in itertools.count():
            latest_chemsys = self.get_latest_chemsys()
            if latest_chemsys:
                with ScratchDir('.') as sd:
                    print("Running {} in {}".format(latest_chemsys, sd))
                    self.run_campaign(latest_chemsys)
            else:
                print("No new campaigns submitted, sleeping for 60 seconds")
                time.sleep(60)
            if num_loops and campaign_num >= num_loops - 1:
                break

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
        s3_client = boto3.client("s3")

        # Get submissions
        submission_prefix = '/'.join([self.campaign, "submit"])
        # TODO: fix 1000 return value limit - MAT-838
        submit_objects = s3_client.list_objects(
            Bucket=CAMD_S3_BUCKET, Prefix=submission_prefix)
        submission_times = {obj['Key'].split('/')[-2]: obj['LastModified']
                            for obj in submit_objects['Contents']
                            if obj['Key'] != "{}/submit/".format(self.campaign)}

        # Get started jobs
        started = get_common_prefixes(CAMD_S3_BUCKET, "/".join([self.campaign, "runs"]))

        # Filter started jobs and then get latest unstarted
        unstarted = list(set(submission_times.keys()) - set(started))
        latest_unstarted = sorted(unstarted, key=lambda x: submission_times[x])

        return latest_unstarted[-1] if latest_unstarted else None


def get_common_prefixes(bucket, prefix):
    """
    Helper function to get common "subfolders" of folders
    in S3

    Args:
        bucket:
        prefix:

    Returns:

    """
    if not prefix.endswith('/'):
        prefix += "/"
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects')
    result = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix)
    return [prefix['Prefix'].split('/')[-2] for prefix in result.search("CommonPrefixes")]


if __name__ == "__main__":
    worker = Worker("proto-dft")
    worker.start()
