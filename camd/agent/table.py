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
        "hull_distance": [0.5, 0.1, 0.15, 0.2],
        "exploit_fraction": [0.4, 0.5, 0.6],
        "regressor": REGRESSOR_PARAMS
    },
]


class HashedParameterArray(object):
    """
    This object is an append-only array which composes
    the IndexedOrderedDict in order to allow for efficient
    lookup of values and indices, the structure is effectively

    {"first_key": 0, "second_key": 1, ..., "latest_key": n}

    such that item inclusion and forwards and backwards
    (i. e. index->key and key->index) are O(1).  Note that
    items in a HashedParameterArray **must be hashable**

    Its getitem behavior defaults to access via the index
    of insertion order, e. g. array[0] returns the first
    inserted object.


    """
    def __init__(self, ordering=None):
        """
        Constructs a HashedParameterArray

        Args:
            ordering (list): initial ordering of hashable
                objects to be stored in the array
        """
        # Construct IndexedOrderedDict
        self._iod = IndexedOrderedDict()
        if ordering:
            self.extend(ordering)

    def __getitem__(self, index):
        return self._iod.keys()[index]

    def __contains__(self, item):
        return item in self._iod.keys()

    def __len__(self):
        return len(self._iod)

    def append(self, item):
        """
        Appends a value to the array

        Args:
            item: item to be inserted

        Returns:
            (int) index of the item appended, if it's found
                returns an existing index, if not, returns
                the index of the inserted (i. e. latest)
                index

        """
        if item in self._iod.keys():
            return self.get_index(item)
        # Otherwise adds item
        value = len(self._iod)
        self._iod[item] = value
        return value

    def extend(self, items):
        """
        Appends multiple items to the array

        Args:
            items ([]): items to be appended

        Returns:
            ([]) indices of appended items in the array

        """
        return [self.append(item) for item in items]

    def get_index(self, item):
        """
        Fetches the index of a given item

        Args:
            item: item to be fetched
        """
        return self._iod[item]


class ParameterTable(object):
    """
    ParameterTable is an object designed to be an
    append-only parameter set for Agent-based parameter
    tuning.  It consists of three attributes which are
    support data structures:

    _parameter_names (HashedParameterArray): an array
        of parameter names which have been added to
        the parameter table
    _parameter_values ({parameter_name: HashedParameterArray(values)}):
        a dict keying the parameter value arrays on the parameter names
    _parameter_sets (HashedParameterArray): an array of tuples
        corresponding to the indices of the parameter name and
        the parameter value, e. g. (0, 3, 1, 4) corresponds to
        a parameterization of the 4th value of the first parameter
        and the fifth value of the second parameter, since arrays
        are zero-indexed.

    Note that configurations to be appended have a syntax in which
    each key is the parameter name, and each value is a list of
    options, for example,

    {"fit_intercept": [True, False], "normalize": [True, False]}

    Will result in a parameter table of "fit_intercept" and "normalize"
    parameters with two values apiece.  Entries in the _parameter_table
    rows will look like (0, 0, 1, 0), (0, 1, 1, 0), (0, 0, 1, 1)
    and (0, 1, 1, 1).  These data structures are easily hashable
    and allow for efficient lookup of prior parameter sets.

    Note that ParameterTables **can be nested**.  If a configuration
    contains lists of dictionaries as values, those dictionaries
    will be cast to parameter tables themselves.

    Lastly, note that getitem, len, and iteration behavior all
    correspond to that behavior of the parameter set array.
    """

    def __init__(self, configs=None):
        """
        Constructs the ParameterTable, using initial
        configurations.

        Args:
            configs ([{}]): list of dictionaries corresponding
                to initial configuration for the parameter table
        """
        self._parameter_names = HashedParameterArray()
        self._parameter_values = {}
        self._parameter_sets = HashedParameterArray()
        for config in configs:
            self.append(config)

    def append(self, config):
        """
        Append operation for the Parameter Table, ingests
        a configuration, i. e. checks for new parameter names
        parameter values, and parameter set options.

        Args:
            config ({}): configuration for parameter sets
                to be added (see above for configuration
                schema).

        Returns:
            (list) of parameter sets represented by the
                configuration, note that these may not
                all be new parameter configurations

        """
        flattened_params = []
        for parameter_name, value_list in sorted(config.items()):
            # New parameter
            if parameter_name not in self._parameter_names:
                self._parameter_names.append(parameter_name)
                if not isinstance(value_list, (list, HashedParameterArray, ParameterTable)):
                    raise ValueError("Values must be a list")
                if isinstance(value_list[0], dict):
                    value_list = ParameterTable(value_list)
                    self._parameter_values[parameter_name] = value_list
                # # Still thinking here,
                # # Just gonna keep tuples for now
                # elif isinstance(value_list[0], list):
                #     pass
                else:
                    self._parameter_values[parameter_name] = HashedParameterArray(value_list)
            # Updating existing parametere
            else:
                if isinstance(value_list[0], dict):
                    value_list = self._parameter_values[parameter_name].extend(value_list)
                else:
                    self._parameter_values[parameter_name].extend(value_list)

            # Accumulate parameter sets
            flattened_params.append(
                [(self._parameter_names.get_index(parameter_name),
                  self._parameter_values[parameter_name].get_index(value))
                 for value in value_list]
            )
        # Accumulate and extend all_parameter_sets
        all_param_sets = []
        total = np.prod([len(el) for el in flattened_params])
        for param_set in tqdm(itertools.product(*flattened_params), total=total):
            tupled = tuple(itertools.chain.from_iterable(param_set))
            all_param_sets.append(tupled)
            self._parameter_sets.append(tupled)
        return all_param_sets

    def extend(self, configs):
        """
        Multi-append, i. e. append multiple configurations in sequence.

        Args:
            configs ([{}]): configurations to append

        Returns:
            list(tuple) - list of tuples that correspond to
                configurations

        """
        all_param_sets = []
        for config in configs:
            all_param_sets.extend(self.append(config))
        return all_param_sets

    def get_index(self, parameter_set):
        """
        Gets the index of a particular parameter set

        Args:
            parameter_set (tuple): unhydrated parameter
                tuple

        Returns:
            (int) index of the row in the parameter set

        """
        return self._parameter_sets.get_index(parameter_set)

    def hydrate_pair(self, param_index, value_index, construct_object=False):
        """
        Hydrates a param index and value index into a
        {param_name: param_value} representation, primarily
        a helper function.  Also will recurse into a sub row
        in a nested parameter table.  Supports object
        construction with the @class parameter which will
        construct objects in sub-parameter table hydration.

        Args:
            param_index (int): index of the parameter name in _parameter_names
            value_index (int): index of the parameter value
            construct_object (bool): whether or not to construct an object
                in a nested parameter table

        Returns:
            (dict) - corresponding to {parameter_name: parameter_value}

        """
        name = self._parameter_names[param_index]
        values = self._parameter_values[name]
        if isinstance(values, ParameterTable):
            sub_row = values[value_index]
            return {name: values.hydrate_row(sub_row, construct_object=construct_object)}
        else:
            return {name: values[value_index]}

    def hydrate(self, parameter_set, construct_object=False):
        """
        Hydrates an entire row into a dict or object representation
        of the parameter set.

        Args:
            parameter_set (tuple): tuple of name and value indices
            construct_object (bool): whether or not to use object
                construction according the @class parameter in
                a parameter set.

        Returns:
            (dict) or object corresponding to the parameter set

        """
        hydrated = {}

        # Group into pairs and update sequentially
        for param_index, value_index in zip(parameter_set[0::2], parameter_set[1::2]):
            hydrated.update(self.hydrate_pair(
                param_index, value_index, construct_object=construct_object))
        if construct_object:
            class_path = hydrated.get("@class")
            if class_path is not None:
                del hydrated['@class']
                constructor = load_class(class_path)
                hydrated = constructor(**hydrated)

        return hydrated

    def hydrate_index(self, index, construct_object=False):
        """
        Similar to hydrate, but hydrates at an index, rather than
        at a parameter set representation

        Args:
            index (int): index of position in the parameter_set array
            construct_object (bool): whether or not to construct
                object in hydration

        Returns:
            (dict) or object corresponding to parameter set index

        """
        return self.hydrate_row(self[index], construct_object=construct_object)



    def __len__(self):
        return len(self._parameter_sets)

    def __iter__(self):
        return iter(self._parameter_sets)

    def __getitem__(self, item):
        return self._parameter_sets[item]

    def __repr__(self):
        return "\n".join([
            self.__class__.__name__,
            "Parameter names: {}".format([name for name in self._parameter_names]),
            "Number of rows: {}".format(len(self._parameter_sets))
            ]
        )

    def __str__(self):
        return "\n".join([
            self.__class__.__name__,
            "Parameter names: {}".format([name for name in self._parameter_names]),
            "Number of rows: {}".format(len(self._parameter_sets))
            ]
        )


def load_class(class_path):
    """
    Load and return the class from the given module.

    Args:
        class_path (str): full path to class to be loaded, e. g.
            sklearn.linear_model.LinearRegression

    Returns:
        class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    mod = __import__(module_path, globals(), locals(), [class_name], 0)
    return getattr(mod, class_name)


# Some things to test
# Whether prior iteration always has same first row set
# Synchronicity of order and value for parameter lists/tables


if __name__ == "__main__":
    first = ParameterTable(AGENT_PARAMS)
    first.hydrate_index(3)
    first.hydrate_index(3, construct_object=True)
