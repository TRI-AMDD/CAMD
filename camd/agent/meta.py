#  Copyright (c) 2019 Toyota Research Institute
"""
This module implements agent-tools for meta-analysis of
other agents
"""


import numpy as np
from taburu.table import ParameterTable
from camd.agent.base import HypothesisAgent


REGRESSOR_PARAMS = [
        {
            "@class": ["sklearn.linear_model.LinearRegression"],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        {
            "@class": ["sklearn.ensemble.RandomForestRegressor"],
            "n_estimators": [100],
            "max_features": list(np.arange(0.05, 1.01, 0.05)),
            "min_samples_split": list(range(2, 21)),
            "min_samples_leaf": list(range(1, 21)),
            "bootstrap": [True, False]
        },
        {
            "@class": ["sklearn.neural_network.MLPRegressor"],
            "hidden_layer_sizes": [
                # I think there's a better way to support this, but need to think
                (80, 50),
                (84, 55),
                (87, 60),
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        },
    ]


AGENT_PARAMS = [
    {
        "@class": ["camd.agent.agents.QBCStabilityAgent"],
        "n_query": [4, 6, 8],
        "n_members": list(range(2, 5)),
        "hull_distance": list(np.arange(0.05, 0.21, 0.05)),
        "training_fraction": [0.4, 0.5, 0.6],
        "regressor": REGRESSOR_PARAMS
    },
    {
        "@class": ["camd.agent.agents.AgentStabilityML5"],
        "n_query": [4, 6, 8],
        "hull_distance": [0.05, 0.1, 0.15, 0.2],
        "exploit_fraction": [0.4, 0.5, 0.6],
        "regressor": REGRESSOR_PARAMS
    },
]


class RandomMetaAgent(HypothesisAgent):
    def __init__(self, agent_pool, candidate_data=None,
                 seed_data=None, n_query=1):
        """
        Args:
            agent_pool (ParameterTable): parameter table corresponding
                to serialized agents in order to serialize on the fly
            candidate_data (DataFrame): candidate data dataframe
            seed_data (DataFrame): seed data data frame
            n_query (int): number of hypotheses to generate
        """
        self.agent_pool = agent_pool
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        super(RandomMetaAgent, self).__init__()

    def get_hypotheses(self, candidate_data=None, seed_data=None):
        """
        Acquires random agents and deserializes the results

        Args:
            candidate_data (DataFrame): candidate data dataframe
            seed_data (DataFrame): seed data data frame

        Returns:
            (DataFrame): dataframe of hypotheses with the "agent"
                field populated

        """
        hypotheses = self.candidate_data.sample(self.n_query)
        hypotheses['agent'] = [self.agent_pool.hydrate(ind)
                               for ind in hypotheses.index]
        return hypotheses


if __name__ == "__main__":
    first = ParameterTable(AGENT_PARAMS)
    first.hydrate_index(3)
    first.hydrate_index(3, construct_object=True)
