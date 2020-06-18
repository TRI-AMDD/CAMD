#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
import os
import boto3
import json

from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET, CAMD_CACHE
from camd.campaigns.base import Campaign
from camd.agent.stability import AgentStabilityML5
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler


class MultiFidelityCampaignTest(unittest.TestCase):
    def setUp():
        self.data = pd.read_csv(os.path.join(CAMD_CACHE, "featurized_brgoch_data.csv"), index_col=0)
        
    def test_agent(self):
        pass
    
    def test_analyzer(self):
        # Split dataframe
        # pick some candidates explicitly
        # Run analyzer on seed, candidates
        # Assert that analyzer gives us expected metric for discovery
        
        # What percent of top threshold candidates have been discovered
        
        # How many candidates discovered in last iteration
        # Weighted experiment value
        pass
    
    def test_experiment(self):
        pass
        

if __name__ == '__main__':
    unittest.main()
