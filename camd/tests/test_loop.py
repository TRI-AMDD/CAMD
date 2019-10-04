#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
import os
import boto3
import json

from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET
from camd.loop import Loop
from camd.agent.stability import AgentStabilityML5
from camd.analysis import AnalyzeStability_mod
from camd.experiment.base import ATFSampler


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="{}".format("test")).delete()


class LoopTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()

    def test_sync(self):
        with ScratchDir('.'):
            df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
            experiment_params = {"dataframe": df}

            # Construct and start loop
            new_loop = Loop(df, AgentStabilityML5, ATFSampler,
                            AnalyzeStability_mod, agent_params={}, create_seed=True,
                            analyzer_params={}, experiment_params=experiment_params,
                            s3_prefix="test")
            new_loop.initialize()
        s3 = boto3.resource('s3')
        obj = s3.Object(CAMD_S3_BUCKET, "test/iteration.json")
        loaded = json.loads(obj.get()['Body'].read())
        self.assertEqual(loaded, 0)

    def test_generate_final_report(self):
        with ScratchDir('.'):
            df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
            experiment_params = {"dataframe": df}

            # Construct and start loop
            new_loop = Loop(df, AgentStabilityML5, ATFSampler,
                            AnalyzeStability_mod, agent_params={}, create_seed=True,
                            analyzer_params={}, experiment_params=experiment_params,
                            )
            new_loop.generate_report_plot(
                "report.png", os.path.join(CAMD_TEST_FILES, "report.log"))
            self.assertTrue(os.path.isfile("report.png"))


if __name__ == '__main__':
    unittest.main()
