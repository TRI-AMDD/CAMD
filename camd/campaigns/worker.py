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
                with ScratchDir('.'):
                    self.run_campaign(latest_chemsys)
            else:
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
        submit_objects = s3_client.list_objects(
            Bucket=CAMD_S3_BUCKET, Prefix=submission_prefix)
        submission_times = {obj['Key'].split('/')[-2]: obj['LastModified']
                            for obj in submit_objects['Contents']}

        # Get started jobs
        start_prefix = '/'.join([self.campaign, "runs"])
        all_objects = s3_client.list_objects_v2(
            Bucket=CAMD_S3_BUCKET, Prefix=start_prefix).get("Contents", [])
        keys = [obj.get('Key') for obj in all_objects]
        pattern = re.compile("runs\/([A-Za-z\-]+)\/")
        started = set(pattern.findall(''.join(keys)))

        # Filter started jobs and then get latest unstarted
        unstarted = list(set(submission_times.keys()) - started)
        latest_unstarted = sorted(unstarted, key=lambda x: submission_times[x])

        return latest_unstarted[-1] if latest_unstarted else None


if __name__ == "__main__":
    worker = Worker("proto-dft")
    worker.start()
