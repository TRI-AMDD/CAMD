#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import traceback
import pandas as pd
import os

from datetime import datetime
from monty.serialization import dumpfn
from camd.domain import StructureDomain, heuristic_setup
from camd.agent.agents import AgentStabilityAdaBoost
from camd.agent.base import RandomAgent
from camd.experiment import ATFSampler
from camd.loop import Loop
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET
from camd.utils.s3 import s3_sync

from camd.analysis import AnalyzeStability, FinalizeQqmdCampaign
from camd.experiment.dft import OqmdDFTonMC1
from sklearn.neural_network import MLPRegressor
import pickle


__version__ = "2019.09.16"


# TODO: abstract campaign?

def run_proto_dft_campaign(chemsys):
    """

    Args:
        chemsys (str): chemical system for the campaign

    Returns:
        (bool): True if run exits

    """
    s3_prefix = "proto-dft/runs/{}".format(chemsys)

    # Initialize s3
    dumpfn({"started": datetime.now().isoformat(),
            "version": __version__}, "start.json")
    s3_sync(s3_bucket=CAMD_S3_BUCKET, s3_prefix=s3_prefix, sync_path='.')

    try:
        # Get structure domain
        element_list = chemsys.split('-')
        g_max, charge_balanced = heuristic_setup(element_list)
        domain = StructureDomain.from_bounds(
            element_list, charge_balanced=charge_balanced,
            n_max_atoms=20, **{'grid': range(1, g_max)})
        candidate_data = domain.candidates()
        structure_dict = domain.hypo_structures_dict

        # Dump structure/candidate data
        with open('candidate_data.pickle', 'wb') as f:
            pickle.dump(candidate_data, f)
        with open('structure_dict.pickle', 'wb') as f:
            pickle.dump(structure_dict, f)

        # Set up agents and loop parameters
        agent = AgentStabilityAdaBoost
        agent_params = {
            'ml_algorithm': MLPRegressor,
            'ml_algorithm_params': {'hidden_layer_sizes': (84, 50)},
            'n_query': 10,
            'hull_distance': 0.2,  # Distance to hull to consider a finding as discovery (eV/atom)
            'exploit_fraction': 1.0,  # Fraction to exploit (rest will be explored -- randomly picked)
            'uncertainty': True,
            'alpha': 0.5,
            'n_estimators': 20
        }
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.2}  # analysis criterion (need not be exactly same as agent's goal)
        experiment = OqmdDFTonMC1
        experiment_params = {'structure_dict': structure_dict, 'candidate_data': candidate_data, 'timeout': 30000}
        experiment_params.update({'timeout': 30000})
        finalizer = FinalizeQqmdCampaign
        finalizer_params = {'hull_distance': 0.2}
        n_max_iter = n_max_iter_heuristics(len(candidate_data), 10)

        # Construct and start loop
        new_loop = Loop(
            candidate_data, agent, experiment, analyzer, agent_params=agent_params,
            analyzer_params=analyzer_params, experiment_params=experiment_params,
            finalizer=finalizer, finalizer_params=finalizer_params, heuristic_stopper=5,
            s3_prefix="proto-dft/runs/{}".format(chemsys))
        new_loop.auto_loop_in_directories(
            n_iterations=n_max_iter, timeout=10, monitor=True,
            initialize=True, with_icsd=True)
    except Exception as e:
        error_msg = {"error": "{}".format(e),
                     "traceback": traceback.format_exc()}
        dumpfn(error_msg, "error.json")
        dumpfn({"status": "error"}, "job_status.json")
        s3_sync(s3_bucket=CAMD_S3_BUCKET, s3_prefix=s3_prefix, sync_path='.')

    return True


def run_atf_campaign(chemsys):
    """
    A very simple test campaign

    Returns:
        True

    """
    s3_prefix = "oqmd-atf/runs/{}".format(chemsys)
    df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
    n_seed = 200  # Starting sample size
    n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
    agent = RandomAgent
    agent_params = {'hull_distance': 0.05, 'N_query': n_query}
    analyzer = AnalyzeStability
    analyzer_params = {'hull_distance': 0.05}
    experiment = ATFSampler
    experiment_params = {'dataframe': df}
    candidate_data = df
    new_loop = Loop(candidate_data, agent, experiment, analyzer,
                    agent_params=agent_params, analyzer_params=analyzer_params,
                    experiment_params=experiment_params, create_seed=n_seed,
                    s3_prefix=s3_prefix)

    new_loop.initialize()

    for _ in range(3):
        new_loop.run()

    return True


def n_max_iter_heuristics(n_data, n_query, low_bound=5, up_bound=20):
    """
    Helper method to define maximum number of iterations for a given campaign.
    This is based on the empirical evidence in various systems >90% of stable
        materials are identified when 25% of candidates are tested. We also enforce
        upper and lower bounds of 20 and 5 to avoid edge cases with too many or too few
        calculations to run.
    Args:
        n_data (int): number of data points in candidate space
        n_query (int): number of queries allowed in each iteration
        low_bound (int): lower bound allowed for n_max_iter
        up_bound (int): upper bound allowed for n_max_ite
    Returns:
        maximum number of iterations as integer
    """
    _target = round(n_data*0.25/n_query)
    if _target<low_bound:
        return low_bound
    else:
        return min(_target, up_bound)
