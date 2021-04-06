#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides objects related to the discovery of
new crystal structures using structural domains.
"""

import logging
import os
import pandas as pd
from datetime import datetime
from monty.serialization import dumpfn
from camd import CAMD_TEST_FILES
from camd.agent.base import RandomAgent
from camd.domain import StructureDomain, heuristic_setup
from camd.agent.stability import AgentStabilityAdaBoost
from camd.campaigns.base import Campaign
from camd import CAMD_S3_BUCKET, __version__
from camd.utils.data import load_dataframe, s3_sync, s3_key_exists, \
    upload_s3_file, download_s3_file
from camd.analysis import StabilityAnalyzer
from camd.experiment.dft import OqmdDFTonMC1
from camd.experiment.base import ATFSampler
from sklearn.neural_network import MLPRegressor
from watchtower import CloudWatchLogHandler


class ProtoDFTCampaign(Campaign):
    """
    Subclass of Campaign which implements custom methods
    and factories for constructing prototype-generation
    stability campaigns for materials discovery with DFT
    experiments
    """
    @classmethod
    def from_chemsys(cls, chemsys, prefix="proto-dft-2/runs",
                     n_max_atoms=20, agent=None, analyzer=None,
                     experiment=None, log_file="campaign.log",
                     cloudwatch_group="/camd/worker/dev/"):
        """
        Class factory method for constructing campaign from
        chemsys.

        Args:
            chemsys (str): chemical system for the campaign
            prefix (str): prefix for s3
            n_max_atoms (int): number of maximum atoms
            agent (Agent): agent for stability campaign
            analyzer (Analyzer): analyzer for stability campaign
            experiment (Agent): experiment for stability campaign
            log_file (str): log filename
            cloudwatch_group (str): cloudwatch group to log to

        Returns:
            (ProtoDFTCampaign): Standard proto-dft campaign from
                the chemical system

        """
        logger = logging.Logger("camd")
        logger.setLevel("INFO")
        file_handler = logging.FileHandler(log_file)
        cw_handler = CloudWatchLogHandler(
            log_group=cloudwatch_group,
            stream_name=chemsys
        )
        logger.addHandler(file_handler)
        logger.addHandler(cw_handler)
        logger.addHandler(logging.StreamHandler())

        logger.info("Starting campaign factory from_chemsys {}".format(
            chemsys)
        )
        s3_prefix = "{}/{}".format(prefix, chemsys)

        # Initialize s3
        dumpfn({"started": datetime.now().isoformat(),
                "version": __version__}, "start.json")
        s3_sync(s3_bucket=CAMD_S3_BUCKET, s3_prefix=s3_prefix, sync_path='.')

        # Get structure domain
        # Check cache
        cache_key = "protosearch_cache/v1/{}/{}/candidate_data.pickle".format(chemsys, n_max_atoms)
        # TODO: create test of isfile
        if s3_key_exists(bucket=CAMD_S3_BUCKET, key=cache_key):
            logger.info("Found cached protosearch domain.")
            download_s3_file(cache_key, CAMD_S3_BUCKET, "candidate_data.pickle")
            candidate_data = pd.read_pickle("candidate_data.pickle")
            logger.info("Loaded cached {}.".format(cache_key))
        else:
            logger.info("Generating domain with max {} atoms.".format(n_max_atoms))
            element_list = chemsys.split('-')
            max_coeff, charge_balanced = heuristic_setup(element_list)
            domain = StructureDomain.from_bounds(
                element_list, charge_balanced=charge_balanced,
                n_max_atoms=n_max_atoms, **{'grid': range(1, max_coeff)})
            candidate_data = domain.candidates()
            logger.info("Candidates generated")
            candidate_data.to_pickle("candidate_data.pickle")
            upload_s3_file(cache_key, CAMD_S3_BUCKET, "candidate_data.pickle")
            logger.info("Cached protosearch domain at {}.".format(cache_key))

        # Dump structure/candidate data

        s3_sync(s3_bucket=CAMD_S3_BUCKET, s3_prefix=s3_prefix, sync_path='.')

        # Set up agents and loop parameters
        agent = agent or AgentStabilityAdaBoost(
            model=MLPRegressor(hidden_layer_sizes=(84, 50)),
            n_query=10,
            hull_distance=0.2,
            exploit_fraction=1.0,
            uncertainty=True,
            alpha=0.5,
            diversify=True,
            n_estimators=20
        )
        analyzer = analyzer or StabilityAnalyzer(hull_distance=0.2)
        experiment = experiment or OqmdDFTonMC1(timeout=30000, prefix_append="proto-dft")
        seed_data = load_dataframe("oqmd1.2_exp_based_entries_featurized_v2")

        # Load cached experiments
        logger.info("Loading cached experiments")
        cached_experiments = experiment.fetch_cached(candidate_data)
        logger.info("Found {} experiments.".format(len(cached_experiments)))
        if len(cached_experiments) > 0:
            summary, seed_data = analyzer.analyze(cached_experiments, seed_data)
            # Remove cached experiments from candidate_data
            candidate_space = candidate_data.index.difference(
                cached_experiments.index, sort=False
            ).tolist()
            candidate_data = candidate_data.loc[candidate_space]
            logger.info("Cached experiments added to seed.")

        # Construct and start loop
        return cls(
            candidate_data=candidate_data, agent=agent, experiment=experiment,
            analyzer=analyzer, seed_data=seed_data,
            heuristic_stopper=5, s3_prefix=s3_prefix,
            logger=logger
        )

    @classmethod
    def from_chemsys_high_quality(cls, chemsys, **kwargs):
        """
        Factory method for generating higher-tier campaigns,
        i.e. with longer walltime, higher quality batch queue,
        and higher number of maximum atoms in candidates

        Args:
            chemsys (str): chemical system for the campaign

        Returns:
            (ProtoDFTCampaign) for campaign corresponding to chemical system

        """
        experiment = OqmdDFTonMC1(timeout=120000, batch_queue="oqmd_prod",
                                  prefix_append="proto-dft-high")
        return cls.from_chemsys(
            chemsys, n_max_atoms=40, experiment=experiment,
            prefix="proto-dft-high/runs", **kwargs)

    def autorun(self):
        """
        Method for running this campaign automatically

        Returns:
            None

        """
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
        """
        Runs campaign with standard parameters
        Returns:
            None
        """
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
