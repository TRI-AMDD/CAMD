#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
import os
import boto3
import json

from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET
from camd.campaigns.base import Campaign
from camd.agent.stability import AgentStabilityML5
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="{}".format("test")).delete()


class CampaignTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()

    @unittest.skipUnless(CAMD_S3_BUCKET, "CAMD S3 Bucket not set")
    def test_sync(self):
        with ScratchDir('.'):
            df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))

            # Construct and start campaign
            new_campaign = Campaign(df, AgentStabilityML5(), ATFSampler(df),
                                    StabilityAnalyzer(), create_seed=10,
                                    s3_prefix="test")
            new_campaign.auto_loop(n_iterations=3, save_iterations=True, initialize=True)
        # Test iteration read
        s3 = boto3.resource('s3')
        obj = s3.Object(CAMD_S3_BUCKET, "test/iteration.json")
        loaded = json.loads(obj.get()['Body'].read())
        self.assertEqual(loaded, 2)

        # Test save directories
        for iteration in [-1, 0, 1, 2]:
            obj = s3.Object(CAMD_S3_BUCKET, f"test/{iteration}/iteration.json")
            loaded = json.loads(obj.get()['Body'].read())
            self.assertEqual(loaded, iteration)


if __name__ == '__main__':
    unittest.main()
