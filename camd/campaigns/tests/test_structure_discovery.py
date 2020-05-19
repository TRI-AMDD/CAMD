#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import boto3
import os

os.environ['CAMD_S3_BUCKET'] = 'camd-test'
from camd.campaigns.structure_discovery import ProtoDFTCampaign
from camd.agent.stability import AgentStabilityML5
from camd import CAMD_S3_BUCKET
from monty.tempfile import ScratchDir

CAMD_DFT_TESTS = os.environ.get("CAMD_DFT_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_DFT_TESTS to run long tests"


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="proto-dft-2/runs/Si").delete()
    bucket.objects.filter(Prefix="proto-dft-2/submit/Si").delete()


class ProtoDFTCampaignTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()

    @unittest.skipUnless(CAMD_DFT_TESTS, SKIP_MSG)
    def test_simple_dft(self):
        with ScratchDir('.'):
            campaign = ProtoDFTCampaign.from_chemsys("Si")
            # Nerf agent a bit
            agent = AgentStabilityML5(n_query=2)
            campaign.agent = agent
            campaign.autorun()


if __name__ == '__main__':
    unittest.main()
