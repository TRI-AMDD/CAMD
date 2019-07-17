#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Initial worker to run structure discovery.  WIP.
"""
import time
import boto3
from camd import CAMD_S3_BUCKET
from camd.log import camd_traced
from camd.campaigns.structure_discovery import run_dft_campaign, run_atf_campaign
import itertools


# TODO: set up test bucket, instead of using complicated pathing

@camd_traced
class Worker(object):
    def __init__(self, campaign="structure_discovery_dft", s3_prefix=None):
        self.campaign = campaign
        self.s3_prefix = s3_prefix

    def start(self, num_loops=None):
        """
        Starts the worker, which monitors s3 for new submissions
        and starts runs accordingly.

        Args:
            num_campaigns (int): number of campaigns to run before
                stopping

        Returns:

        """
        for campaign_num in itertools.count():
            latest_chemsys = self.get_latest_chemsys()
            if latest_chemsys:
                self.run_campaign(latest_chemsys)
            time.sleep(60)
            if campaign_num > num_loops + 1:
                break

    def run_campaign(self, chemsys):
        """
        Runs the campaign for a given chemsys

        Args:
            chemsys ([str]):

        Returns:

        """
        s3_prefix = '/'.join([self.s3_prefix, '-'.join(sorted(chemsys))])
        if self.campaign == "structure_discovery":
            run_dft_campaign(chemsys, s3_prefix=s3_prefix)
        elif self.campaign == "random_atf":
            # This is more or less just a test
            run_atf_campaign(s3_prefix=s3_prefix)
        else:
            raise ValueError("Campaign {} is not valid".format(self.campaign))

    def get_latest_chemsys(self):
        s3_client = boto3.client("s3")

        # Get submissions
        submission_prefix = '/'.join([self.s3_prefix, "submit"])
        submit_objects = s3_client.list_objects_v2(
            CAMD_S3_BUCKET, submission_prefix)
        submission_times = {obj['Key'].split('/')[-2]: obj['LastUpdated']
                            for obj in submit_objects}

        # Get started jobs
        start_prefix = '/'.join([self.s3_prefix, "start"])
        started = s3_client.list_objects_v2(CAMD_S3_BUCKET, start_prefix)
        started = started.get("CommonPrefixes")

        # Filter started jobs and then get latest unstarted
        unstarted = list(set(submission_times.keys()) - set(started))
        latest_unstarted = sorted(unstarted, key=lambda x: submission_times[x])

        return latest_unstarted
