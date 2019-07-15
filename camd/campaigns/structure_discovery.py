#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import os

from camd.domain import StructureDomain
from camd.loop import Loop
from camd.agent.agents import QBCStabilityAgent, AgentStabilityML5
from camd.analysis import AnalyzeStability_mod
from camd.experiment.dft import OqmdDFTonMC1
from sklearn.neural_network import MLPRegressor
import pickle


__version__ = "2019.07.15"


CAMD_RUN_LOC = os.environ.get("CAMD_RUN_LOC", ".")


def run_structure_discovery_campaign(chemsys):
    """

    Args:
        chemsys (List): list of elements in which to do
            the chemsys

    Returns:
        (bool): True if run exits

    """
    # Get structure domain
    domain = StructureDomain.from_bounds(
        chemsys, n_max_atoms=12, **{'grid': range(1,3)})
    candidate_data = domain.candidates()
    structure_dict = domain.hypo_structures_dict

    # Dump structure/candidate data
    with open('candidate_data.pickle', 'wb') as f:
        pickle.dump(candidate_data, f)
    with open('structure_dict.pickle', 'wb') as f:
        pickle.dump(structure_dict, f)

    # Set up agents and loop parameters
    agent = AgentStabilityML5  # Query-by-committee agent that operates with maximum expected gain
    agent_params = {
        'ML_algorithm': MLPRegressor,  # We'll use a simple NN regressor
        'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
        'N_query': 10,  # Number of experiments the agent can request in each round.
        'hull_distance': 0.25,  # Distance to hull to consider a finding as discovery (eV/atom)
        'frac': 0.7  # Fraction to exploit
    }
    analyzer = AnalyzeStability_mod
    analyzer_params = {'hull_distance': 0.2}  # analysis criterion (need not be exactly same as agent's goal)
    experiment = OqmdDFTonMC1
    experiment_params = {'structure_dict': structure_dict, 'candidate_data': candidate_data, 'timeout': 20000}
    experiment_params.update({'timeout': 30000})

    # Construct and start loop
    new_loop = Loop(
        candidate_data, agent, experiment, analyzer, agent_params=agent_params,
        analyzer_params=analyzer_params, experiment_params=experiment_params)
    new_loop.auto_loop_in_directories(
        n_iterations=5, timeout=10, monitor=True, initialize=True, with_icsd=True)
    return True
