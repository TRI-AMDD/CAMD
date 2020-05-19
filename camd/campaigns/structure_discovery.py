#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import pandas as pd
import os

from datetime import datetime
from monty.serialization import dumpfn
from camd.domain import StructureDomain, heuristic_setup
from camd.agent.stability import AgentStabilityAdaBoost
from camd.agent.base import RandomAgent
from camd.experiment import ATFSampler
from camd.campaigns.base import Campaign
from camd import CAMD_TEST_FILES, CAMD_S3_BUCKET, __version__
from camd.utils.data import load_dataframe, s3_sync
from camd.analysis import StabilityAnalyzer
from camd.experiment.dft import OqmdDFTonMC1
from sklearn.neural_network import MLPRegressor
import pickle


class ProtoDFTCampaign(Campaign):
    """
    Subclass of Campaign which implements custom methods
    and factories for constructing prototype-generation
    stability campaigns for materials discovery with DFT
    experiments
    """
    @classmethod
    def from_chemsys(cls, chemsys):
        """
        Class factory method for constructing campaign from
        chemsys.

        Args:
            chemsys (str): chemical system for the campaign

        Returns:
            (ProtoDFTCampaign): Standard proto-dft campaign from
                the chemical system

        """
        s3_prefix = "proto-dft-2/runs/{}".format(chemsys)

        # Initialize s3
        dumpfn({"started": datetime.now().isoformat(),
                "version": __version__}, "start.json")
        s3_sync(s3_bucket=CAMD_S3_BUCKET, s3_prefix=s3_prefix, sync_path='.')

        # Get structure domain
        element_list = chemsys.split('-')
        max_coeff, charge_balanced = heuristic_setup(element_list)
        domain = StructureDomain.from_bounds(
            element_list, charge_balanced=charge_balanced,
            n_max_atoms=20, **{'grid': range(1, max_coeff)})
        candidate_data = domain.candidates()

        # Dump structure/candidate data
        with open('candidate_data.pickle', 'wb') as f:
            pickle.dump(candidate_data, f)

        # Set up agents and loop parameters
        agent = AgentStabilityAdaBoost(
            model=MLPRegressor(hidden_layer_sizes=(84, 50)),
            n_query=10,
            hull_distance=0.2,
            exploit_fraction=1.0,
            uncertainty=True,
            alpha=0.5,
            diversify=True,
            n_estimators=20
        )
        analyzer = StabilityAnalyzer(hull_distance=0.2)
        experiment = OqmdDFTonMC1(timeout=30000)
        seed_data = load_dataframe("oqmd1.2_exp_based_entries_featurized_v2")

        # Construct and start loop
        return cls(
            candidate_data=candidate_data, agent=agent, experiment=experiment,
            analyzer=analyzer, seed_data=seed_data,
            heuristic_stopper=5, s3_prefix="proto-dft/runs/{}".format(chemsys)
        )

    def autorun(self):
        n_max_iter = n_max_iter_heuristics(
            len(self.candidate_data), 10)
        self.auto_loop(
            n_iterations=n_max_iter, monitor=True,
            initialize=True, save_iterations=True
        )


class CloudATFCampaign(Campaign):
    """
    Simple subclass for cloud-based ATF, mostly for testing
    """
    @classmethod
    def from_chemsys(cls, chemsys):
        """

        Args:
            chemsys:

        Returns:

        """
        s3_prefix = "oqmd-atf/runs/{}".format(chemsys)
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = RandomAgent(n_query=n_query)
        analyzer = StabilityAnalyzer(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df)
        candidate_data = df
        return cls(candidate_data, agent, experiment, analyzer,
                   create_seed=n_seed, s3_prefix=s3_prefix)

    def autorun(self):
        self.auto_loop(initialize=True, n_iterations=3)
        return True


def n_max_iter_heuristics(n_data, n_query, low_bound=5, up_bound=20):
    """
    Helper method to define maximum number of iterations for
    a given campaign.  This is based on the empirical evidence
    in various systems >90% of stable materials are identified
    when 25% of candidates are tested. We also enforce upper
    and lower bounds of 20 and 5 to avoid edge cases with too
    many or too few calculations to run.

    Args:
        n_data (int): number of data points in candidate space
        n_query (int): number of queries allowed in each iteration
        low_bound (int): lower bound allowed for n_max_iter
        up_bound (int): upper bound allowed for n_max_ite

    Returns:
        maximum number of iterations as integer

    """
    _target = round(n_data * 0.25/n_query)
    if _target < low_bound:
        return low_bound
    else:
        return min(_target, up_bound)
