#  Copyright (c) 2019 Toyota Research Institute
"""
This module is intended to create a parameter table associated
with agent testing, e. g. to categorize Agents via numerical
vectors
"""
from camd.agent.stability import QBCStabilityAgent, AgentStabilityML5, \
    GaussianProcessStabilityAgent, SVGProcessStabilityAgent, \
    BaggedGaussianProcessStabilityAgent, AgentStabilityAdaBoost

import abc

ML_ALGORITHM_PARAMETER_TABLE = [

]

AGENT_PARAMETER_TABLE = [
    [QBCStabilityAgent, ML_ALGORITHM_PARAMETER_TABLE],
    [AgentStabilityML5, ML_ALGORITHM_PARAMETER_TABLE],
    [GaussianProcessStabilityAgent, ["alpha", float]]

]


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
        for parameter_name, parameter_type in cls.PARAMETER_LIST:
            if isinstance(parameter_type, VectorParameterized):
                parameters.update(
                    {parameter_name: parameter_type.from_vector(vector)}
                )
            else:
                parameters.update(
                    {parameter_name: parameter_type(vector.pop(0))}
                )
        return cls(**parameters)
