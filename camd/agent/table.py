#  Copyright (c) 2019 Toyota Research Institute
"""
This module is intended to create a parameter table associated
with agent testing, e. g. to categorize Agents via numerical
vectors
"""

import numpy as np
import itertools
from indexed import IndexedOrderedDict
from tqdm import tqdm


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


class HashedParameterArray(object):
    def __init__(self, ordering):
        # Construct IndexedOrderedDict
        self._iod = IndexedOrderedDict()
        for n, value in enumerate(ordering):
            self._iod[value] = n

    def __getitem__(self, index):
        return self._iod.keys()[index]

    def __contains__(self, item):
        return item in self._iod.keys()

    # Maybe this should be push or something?
    def append(self, item):
        if item in self._iod.keys():
            return self._iod[item]
        # Otherwise adds item
        self._iod[item] = len(self._iod)

    def extend(self, items):
        for item in items:
            self.append(item)


class ParameterTable(object):
    def __init__(self, configs=None):
        self._parameter_names = []
        self._parameter_values = {}
        self._parameter_table = IndexedOrderedDict()
        for config in configs:
            self.append(config)

    def append(self, config):
        flattened_params = []
        for parameter_name, value_list in sorted(config.items()):
            # Update the parameter vectors
            if parameter_name in self._parameter_names:
                if isinstance(value_list[0], dict):
                    for value in value_list:
                        self._parameter_values[parameter_name].append(value)
                # Check if any new values, if so append to end of list
                current_values = set(self._parameter_values[parameter_name])
                for value in value_list:
                    if isinstance(value, dict):
                        # This should update the nested parameter table
                        self._parameter_values[parameter_name].append(value)
                    if value not in current_values:
                        self._parameter_values[parameter_name].append(value)
                # TODO: fix recursion here
            else:
                self._parameter_names.append(parameter_name)
                # Maybe I should hash on the int?
                if isinstance(value_list[0], dict):
                    self._parameter_values.update(
                        {parameter_name: ParameterTable(configs=value_list)}
                    )
                elif isinstance(value_list[0], list):
                    pass  # Deal with lists?
                else:
                    self._parameter_values.update(
                        {parameter_name: value_list}
                    )
            # Cast name, value list to vectorized
            # I think this lookup could be improved
            flattened_params.append(
                [(self._parameter_names.index(parameter_name),
                  self._parameter_values[parameter_name].index(value))
                 for value in value_list]
            )
        # Accumulate and extend all_parameter_sets
        total = np.prod([len(el) for el in flattened_params])
        for param_set in tqdm(itertools.product(*flattened_params), total=total):
            param_set = tuple(itertools.chain.from_iterable(param_set))
            if self._parameter_table.get(param_set) is None:
                # Maybe this should just be 0, but length makes indexing lookup fast
                # if desired
                self._parameter_table[param_set] = len(self._parameter_table)

    def __len__(self):
        return len(self._parameter_table)

    def __iter__(self):
        return iter(self._parameter_table)

    def __getitem__(self, item):
        return self._parameter_table.keys()[item]

    def row_(self, ):
        raise NotImplementedError("Hydration not yet implemented")
        # # First we flatten the dict
        # flattened_params = []
        # for param_name, value_list in sorted(param_config.items()):
        #     # Simple validation
        #     if not isinstance(value_list, list):
        #         raise ValueError("Values in config must be a list.")

        #     # Begin building name
        #     if prefix is not None:
        #         name = "{}.{}".format(prefix, param_name)
        #     else:
        #         name = "{}".format(param_name)

        #     # If values are dicts, recurse into the value list and generate parameter sets
        #     if isinstance(value_list[0], dict):
        #         flattened_params.append(
        #             enumerate_parameters(value_list, prefix=name))
        #     # Early attempt at supporting multi-entry lists
        #     # Probably a bit problematic
        #     elif isinstance(value_list[0], list):
        #         # Naming is a bit wonky here, because the value_list is a list of lists
        #         for n, values in enumerate(value_list):
        #             name += ".{}".format(n)
        #             flattened_params.append([(name, value) for value in values])

        #     else:
        #         flattened_params.append([(name, value) for value in value_list])
        # # Accumulate and extend all_parameter_sets
        # for param_set in itertools.product(*flattened_params):
        #     yield list(itertools.chain.from_iterable(param_set))



if __name__ == "__main__":
    # for param_set in enumerate_parameters(config=agent_params, prefix="agent"):
    #     print(param_set)
