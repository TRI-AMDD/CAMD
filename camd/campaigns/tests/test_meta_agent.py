#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import boto3
import os

os.environ['CAMD_S3_BUCKET'] = 'camd-test'
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.utils.data import load_default_atf_data, get_oqmd_data_by_chemsys, \
    partition_intercomp
from camd.campaigns.meta_agent import MetaAgentCampaign, StabilityCampaignAnalyzer, \
    META_AGENT_PREFIX
from camd.experiment.agent_simulation import LocalAgentSimulation
from monty.tempfile import ScratchDir
from camd.analysis import StabilityAnalyzer
from camd.agent.base import RandomAgent


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix=META_AGENT_PREFIX).delete()


TEST_REGRESSOR_PARAMS = [
    {
        "@class": ["sklearn.linear_model.LinearRegression"],
        "fit_intercept": [True, False],
        "normalize": [True, False]
    },
]


TEST_AGENT_PARAMS = [
    {
        "@class": ["camd.agent.stability.QBCStabilityAgent"],
        "n_query": [10],
        "n_members": list(range(2, 5)),
        "hull_distance": [0.05],
        "training_fraction": [0.4],
        "model": TEST_REGRESSOR_PARAMS
    },
]


RANDOM_TEST_AGENT_PARAMS = [
    {
        "@class": ["camd.agent.base.RandomAgent"],
        "n_query": [2, 3],
    },
]


@unittest.skipUnless(CAMD_S3_BUCKET, "CAMD S3 Bucket not set")
class MetaAgentCampaignTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()

    def test_initialize_and_update(self):
        agent_pool = ParameterTable(TEST_AGENT_PARAMS)
        dataframe = get_oqmd_data_by_chemsys("Fe-O")
        cand, seed = partition_intercomp(dataframe, n_elements=1)
        analyzer = StabilityAnalyzer()
        experiment = LocalAgentSimulation(
            cand, iterations=5,
            analyzer=analyzer, seed_data=seed
        )

        MetaAgentCampaign.reserve(
            name="test_meta_agent", experiment=experiment,
            agent_pool=agent_pool, analyzer=analyzer
        )
        self.assertRaises(ValueError, MetaAgentCampaign.reserve,
                          "test_meta_agent", dataframe, agent_pool, None)

        agent_pool, data, analyzer = MetaAgentCampaign.load_pickled_objects(
            "test_meta_agent"
        )
        self.assertEqual(len(agent_pool), 12)

        MetaAgentCampaign.update_agent_pool(
            "test_meta_agent",
            TEST_AGENT_PARAMS
        )
        agent_pool, _, _ = MetaAgentCampaign.load_pickled_objects(
            "test_meta_agent"
        )
        self.assertEqual(len(agent_pool), 12)

    def test_run(self):
        agent_pool = ParameterTable(RANDOM_TEST_AGENT_PARAMS)
        # Construct experiment
        dataframe = get_oqmd_data_by_chemsys("Fe-O")
        cand, seed = partition_intercomp(dataframe, n_elements=1)
        experiment = LocalAgentSimulation(
            atf_candidate_data=cand, seed_data=seed,
            analyzer=StabilityAnalyzer(), iterations=10,
        )
        analyzer = StabilityCampaignAnalyzer(checkpoint_indices=[2, 5, 10])
        MetaAgentCampaign.reserve(
            name="test_meta_agent", experiment=experiment,
            agent_pool=agent_pool, analyzer=analyzer
        )
        with ScratchDir('.'):
            print("Testing meta agent")
            campaign = MetaAgentCampaign.from_reserved_name(
                "test_meta_agent", meta_agent=RandomAgent(n_query=1),
            )
            campaign.autorun()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
