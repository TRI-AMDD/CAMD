import unittest
import pandas as pd
import os
import boto3
import json
import numpy as np

from monty.tempfile import ScratchDir
from sklearn.svm import SVR
from sklearn import preprocessing

from camd.agent.multi_fidelity import EpsilonGreedyMultiAgent, GPMultiAgent
from camd.experiment.base import ATFSampler
from camd.analysis import MultiAnalyzer
from camd.campaigns.base import Campaign
from camd import CAMD_TEST_FILES


class MultiFidelityInTandemCampaignTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_multi_fidelity_df.csv"),
                                index_col=0)
        self.features = [column for column in self.data if column.startswith("MagpieData")]
        self.seed_data = self.data.head(100)
        self.candidate_data = self.data.drop(self.seed_data.index)

    def test_epsilongreedy_agent(self):
        agent = EpsilonGreedyMultiAgent(candidate_data=self.candidate_data, seed_data=self.seed_data,
                                        features=self.features, target_prop='bandgap', target_prop_val=1.8,
                                        model=SVR(C=10), total_budget=10, highFi_query_frac=0.5)
        hypotheses = agent.get_hypotheses(self.candidate_data, self.seed_data)
        self.assertEqual(type(hypotheses), pd.core.frame.DataFrame)
        self.assertEqual(hypotheses.loc[hypotheses.expt_data==1].shape[0], 5)
        self.assertEqual(hypotheses.shape[0], 10)

    def test_GPLCB_agent(self):
        GP_agent = GPMultiAgent(candidate_data=self.candidate_data, seed_data=self.seed_data,
                             features=self.features, target_prop='bandgap', target_prop_val=1.8,
                             alpha=0.08, unc_percentile=100, total_budget=10)

        hypotheses = GP_agent.get_hypotheses(self.candidate_data, self.seed_data)
        self.assertEqual(type(hypotheses), pd.core.frame.DataFrame)
        self.assertEqual(hypotheses.shape[0], 10)

    def test_analyzer(self):
        sample_candidate_data = self.candidate_data.loc[[199, # good expt
                                                         237, # good expt
                                                         101, # bad expt
                                                         286, # good theory
                                                         296  # bad theory
                                                         ]]
        analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])
        summary, new_seed = analyzer.analyze(new_experimental_results=sample_candidate_data, seed_data=self.seed_data)
        self.assertEqual(
                         (summary['expt_queried'].values[0],
                          summary['total_expt_discovery'].values[0],
                          summary['new_discovery'].values[0],
                          summary['success_rate'].values[0]),
                         (3, 2, 2, 2/3)
                        )

    def test_experiment(self):
        pass

    def test_campaign(self):
        # run the campaign with epsilon greedy agent
        agent = EpsilonGreedyMultiAgent(candidate_data=self.candidate_data, seed_data=self.seed_data,
                                        features=self.features, target_prop='bandgap', target_prop_val=1.8,
                                        model=SVR(C=10), total_budget=10, highFi_query_frac=0.5)
        experiment = ATFSampler(dataframe=self.data)
        analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])
        with ScratchDir('.'):
            campaign = Campaign(candidate_data=self.candidate_data, seed_data=self.seed_data,
                                agent=agent, experiment=experiment, analyzer=analyzer)
            campaign.auto_loop(n_iterations=2, initialize=True)
            self.assertTrue(os.path.isfile('history.pickle'))


if __name__ == '__main__':
    unittest.main()