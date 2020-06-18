#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
import os
import boto3
import json
import numpy as np

from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET, CAMD_CACHE
from camd.campaigns.multi import MultiAnalyzer
from camd.agent.stability import AgentStabilityML5
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler


class MultiFidelityCampaignTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(os.path.join(CAMD_CACHE, "featurized_brgoch_data.csv"), index_col=0)
        
    def test_agent(self):
        pass
    
    def test_analyzer(self):
        # Split dataframe
        seed_data = self.data.sample(frac=0.5, random_state=42)
        candidate_data = self.data.drop(seed_data.index)
        # candidate_data['dist_to_ideal'] = np.abs(1.8-candidate_data['bandgap'])
        # candidate_data = candidate_data.sort_values(by=['dist_to_ideal'])

        # pick some candidates explicitly
        sample_candidate_data = candidate_data.loc[[7092, # Good, exp
                                                    8160, # Good, exp
                                                    7816, # Good, calc
                                                    7684, # Bad calc
                                                    6981  # Bad exp
                                                   ]]

        # Run analyzer on seed, candidates
        analyzer = MultiAnalyzer(target_property='bandgap', property_range=[1.6, 2.0])
        summary, new_seed = analyzer.analyze(
            new_experimental_results=sample_candidate_data, seed_data=seed_data)
        self.assertEqual(summary['discovery_rate'][0], 0.4)
        

        # Assert that analyzer gives us expected metric for discovery
        # What percent of top threshold candidates have been discovered
        
        # How many candidates discovered in last iteration
        # Weighted experiment value
        pass
    
    def test_experiment(self):
        pass
        

if __name__ == '__main__':
    unittest.main()
