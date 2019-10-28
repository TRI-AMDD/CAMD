#  Copyright (c) 2019 Toyota Research Institute
"""
This module is intended to create a parameter table associated
with agent testing, e. g. to categorize Agents via numerical
vectors
"""

import abc
import inspect

import numpy as np

from camd.agent.agents import QBCStabilityAgent, AgentStabilityML5, \
    GaussianProcessStabilityAgent, SVGProcessStabilityAgent, \
    BaggedGaussianProcessStabilityAgent, AgentStabilityAdaBoost

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


class VectorParameterized(metaclass=abc.ABCMeta):
    @property
    @classmethod
    @abc.abstractmethod
    def PARAMETER_LIST(cls):
        return NotImplementedError

    @classmethod
    def from_vector(cls, vector, **other_kwargs):
        """
        Recursively consume the vector
        """
        parameters = other_kwargs
        for parameter_name, parameter_option in cls.PARAMETER_LIST:
            if isinstance(parameter_option, list):
                parameter_option = parameter_option[vector.pop(0)]
                if VectorParameterized.is_superclass(parameter_option):
                    parameters.update(
                        {parameter_name: parameter_option.from_vector(vector)}
                    )
                else:
                    parameters.update(
                        {parameter_name: parameter_option}
                    )
            elif VectorParameterized.is_superclass(parameter_option):
                parameters.update(
                    {parameter_name: parameter_option.from_vector(vector)}
                )
            else:
                parameters.update(
                    {parameter_name: parameter_option(vector.pop(0))}
                )
        return cls(**parameters)

    @classmethod
    def is_superclass(cls, target):
        if not inspect.isclass(target):
            return False
        else:
            return issubclass(target, cls)


# TODO: maximum minimum?
class VariableLengthVectorVP(list, VectorParameterized, metaclass=abc.ABCMeta):
    @property
    @classmethod
    @abc.abstractmethod
    def MAX_LENGTH(cls):
        return NotImplementedError

    PARAMETER_LIST = [
        ('length', int),
        ('vector', list)
    ]

    @classmethod
    def from_vector(cls, vector):
        length = vector.pop(0)
        new = []
        actual_length = min(length, cls.MAX_LENGTH)
        for i in range(actual_length):
            new.append(vector.pop(0))
        return new


class HiddenLayerVP(VariableLengthVectorVP):
    MAX_LENGTH = 5


class LinearRegVP(LinearRegression, VectorParameterized):
    PARAMETER_LIST = [
        ("fit_intercept", [True, False]),
        ("normalize", [True, False])
    ]


class RandomForestRegVP(RandomForestRegressor, VectorParameterized):
    PARAMETER_LIST = [
        ("n_estimators", [50, 75, 100]),
        ("max_features", np.arange(0.05, 1.01, 0.05)),
        ("min_samples_split", range(2, 21)),
        ("min_samples_leaf", range(1, 21)),
        ("bootstrap", [True, False]),
    ]


class MLPRegVP(MLPRegressor, VectorParameterized):
    PARAMETER_LIST = [
        ("hidden_layer_sizes", HiddenLayerVP),
        ("activation", ['identity', 'logistic', 'tanh', 'relu']),
        ("learning_rate", ['constant', 'invscaling', 'adaptive']),
    ]


class ScikitRegVP(VectorParameterized):
    PARAMETER_LIST = [
        ("regressor", [LinearRegVP, RandomForestRegVP, MLPRegVP])
    ]

    def __init__(self, regressor):
        self.regressor = regressor

    def __getattr__(self, item):
        if item == "regressor":
            return item
        else:
            return getattr(self.regressor, item)


class QBCStabilityAgentVP(QBCStabilityAgent, VectorParameterized):
    PARAMETER_LIST = [
        ("n_query", int),
        ("n_members", int),
        ("training_fraction", float),
        ("regressor", [LinearRegVP, RandomForestRegVP, MLPRegVP]),
    ]


class AgentStabilityML5VP(AgentStabilityML5, VectorParameterized):
    PARAMETER_LIST = [
        ("n_query", int),
        ("n_members", int),
        ("training_fraction", float),
        ("regressor", ScikitRegVP),
    ]

class LoopVP(Loop, VectorParameterized):
    PARAMETER_LIST = [
        ("agent", [AgentStabilityML5VP, QBCStabilityAgentVP])
    ]

class AgentVP(VectorParameterized):
    PARAMETER_LIST = [
        ("agent", [QBCStabilityAgentVP, AgentStabilityML5VP])
    ]

    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, item):
        if item == "agent":
            return item
        else:
            return getattr(self.agent, item)


agent_config_table = [
    (),
    (),
]


if __name__ == "__main__":
    test = ScikitRegVP.from_vector([0, 1, 1, 0, 0, 1, 0, 0, 0])
    test_agent = AgentVP.from_vector([0, 1, 1, 1, 0, 1, 1])
