#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
import os
import boto3
import json
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)

from monty.tempfile import ScratchDir
from sklearn.svm import SVR
from sklearn import preprocessing

from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET, CAMD_CACHE
from camd.campaigns.multi import MultiAgent, MultiAnalyzer
from camd.experiment.base import ATFSampler


class MultiFidelityCampaignTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(os.path.join(CAMD_CACHE, "brgoch_unittest_data.csv"), index_col=0)
        self.seed_data = self.data.head(1789)
        self.candidate_data = self.data.drop(self.seed_data.index)
        self.assertEqual((self.data.shape[0], self.seed_data.shape[0], self.candidate_data.shape[0]),
                         (1824, 1789, 35))

    def test_agent_hypotheses(self):

        # # run the agent, make assertion
        agent = MultiAgent(target_property='bandgap', ideal_property_value=1.8,
                 candidate_data=self.candidate_data, seed_data=self.seed_data, n_query=25,
                 model=SVR(C=10), preprocessor=preprocessing.StandardScaler()
                 )
        hypotheses = agent.get_hypotheses(self.candidate_data, self.seed_data)

        self.assertEqual(type(hypotheses), pd.core.frame.DataFrame)
        self.assertEqual(hypotheses.loc[hypotheses.expt_data==1].shape[0], 5)
        self.assertEqual(hypotheses.shape[0], 25)

    def test_analyzer(self):

        sample_candidate_data = self.candidate_data.loc[[1789, # good expt
                                                         1793, # good expt
                                                         1791, # bad expt
                                                         1794, # good theory
                                                         1799  # bad theory
                                                         ]]

        # Run analyzer on seed, candidates
        analyzer = MultiAnalyzer(target_property='bandgap', property_range=[1.6, 2.0])
        summary, new_seed = analyzer.analyze(new_experimental_results=sample_candidate_data, seed_data=self.seed_data)
        self.assertEqual((summary['iteration tpr'].values[0],
                          round(summary['iteration experiment tpr'].values[0] ,2),
                          summary['new_exp_discovery'].values[0],
                          summary['total_exp_discovery'].values[0]),
                         (0.6, 0.67, 2, 70))

    def test_experiment(self):
        pass


if __name__ == '__main__':
    unittest.main()
