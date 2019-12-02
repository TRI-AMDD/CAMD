#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import boto3
import os

os.environ['CAMD_S3_BUCKET'] = 'camd-test'
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.utils.data import load_default_atf_data
from camd.campaigns.meta_agent import initialize_agent_campaign, \
    update_agent_pool, load_agent_pool, run_meta_agent_campaign


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="{}".format("agent_testing")).delete()


TEST_REGRESSOR_PARAMS = [
    {
        "@class": ["sklearn.linear_model.LinearRegression"],
        "fit_intercept": [True, False],
        "normalize": [True, False]
    },
    {
            "@class": ["sklearn.neural_network.MLPRegressor"],
            "hidden_layer_sizes": [
                # I think there's a better way to support this, but need to think
                (84, 50)
            ],
            "activation": ["identity"],
            "learning_rate": ["constant"]
    },
]


TEST_AGENT_PARAMS = [
    {
        "@class": ["camd.agent.agents.QBCStabilityAgent"],
        "n_query": [4],
        "n_members": list(range(2, 5)),
        "hull_distance": [0.05],
        "training_fraction": [0.4],
        "regressor": TEST_REGRESSOR_PARAMS
    },
    {
        "@class": ["camd.agent.agents.AgentStabilityML5"],
        "n_query": [4, 6],
        "hull_distance": [0.05],
        "exploit_fraction": [0.4, 0.5],
        "regressor": TEST_REGRESSOR_PARAMS
    },
]


class MetaAgentTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()

    def test_initialize_and_update(self):
        agent_pool = ParameterTable(TEST_AGENT_PARAMS)
        dataframe = load_default_atf_data()
        initialize_agent_campaign(
            name="test_meta_agent", dataframe=dataframe,
            agent_pool=agent_pool
        )
        self.assertRaises(ValueError, initialize_agent_campaign,
                          "test_meta_agent", dataframe)

        first = load_agent_pool("test_meta_agent")

        update_agent_pool(
            "test_meta_agent",
            TEST_AGENT_PARAMS
        )
        second = load_agent_pool("test_meta_agent")
        self.assertEqual(len(first), len(second))

    def test_run(self):
        agent_pool = ParameterTable(TEST_AGENT_PARAMS)
        dataframe = load_default_atf_data()
        initialize_agent_campaign(
            name="test_meta_agent", dataframe=dataframe,
            agent_pool=agent_pool
        )
        run_meta_agent_campaign("test_meta_agent")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
