#  Copyright (c) 2019 Toyota Research Institute
"""
This module is intended to create a parameter table associated
with agent testing, e. g. to categorize Agents via numerical
vectors
"""

import numpy as np
import itertools


sklearn_regressor_params = [
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
                [80, 84],
                [50, 55]
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        },
    ]


agent_params = [
    {
        "@class": ["camd.agent.QBCStabilityAgent"],
        "n_query": [4, 6, 8],
        "n_members": list(range(2, 5)),
        "hull_distance": list(np.arange(0.05, 0.21, 0.05)),
        "training_fraction": [0.4, 0.5, 0.6],
        "regressor": sklearn_regressor_params
    },
    {
        "@class": ["camd.agent.AgentStabilityML5"],
        "n_query": [4, 6, 8],
        "hull_distance": [0.5, 0.1, 0.15, 0.2],
        "exploit_fraction": [0.4, 0.5, 0.6],
        "regressor": sklearn_regressor_params
    },
    {
        "@class": ["camd.agent.AgentStabilityML5"],
        "hull_distance": [0.5, 0.1, 0.15, 0.2],
        "n_query": [4, 6, 8],
        "training_fraction": [0.4, 0.5, 0.6],
        "regressor": sklearn_regressor_params
    }
]


def enumerate_parameters(config, prefix=None):
    all_parameter_sets = []
    for param_config in config:
        # First we flatten the dict
        flattened_params = []
        for param_name, value_list in sorted(param_config.items()):
            # Simple validation
            if not isinstance(value_list, list):
                raise ValueError("Values in config must be a list.")

            # Begin building name
            if prefix is not None:
                name = "{}.{}".format(prefix, param_name)
            else:
                name = "{}".format(param_name)

            # If values are dicts, recurse into the value list and generate parameter sets
            if isinstance(value_list[0], dict):
                flattened_params.append(
                    enumerate_parameters(value_list, prefix=name))
            # Early attempt at supporting multi-entry lists
            # Probably a bit problematic
            elif isinstance(value_list[0], list):
                # Naming is a bit wonky here, because the value_list is a list of lists
                for n, values in enumerate(value_list):
                    name += ".{}".format(n)
                    flattened_params.append([(name, value) for value in values])

            else:
                flattened_params.append([(name, value) for value in value_list])
        # Accumulate and extend all_parameter_sets
        for param_set in itertools.product(*flattened_params):
            yield list(itertools.chain.from_iterable(param_set))


if __name__ == "__main__":
    iter = 0
    for param_set in enumerate_parameters(config=agent_params, prefix="agent"):
        print(param_set)
        iter += 1
        if iter > 500000:
            break
