#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module contains basic campaign functionality.  Objects
and logic in this module should be very generic and not
constrained to a particular mode of materials discovery.

Furthermore, the "Campaign" logic should be kept as simple
as possible.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import shutil
import logging

from monty.json import MSONable
from camd.utils.data import s3_sync
from camd import CAMD_S3_BUCKET
from camd.agent.base import RandomAgent


class Campaign(MSONable):
    """
    Campaign provides a sequential, workflow-like capability where an
    Agent iterates over a candidate space to choose and execute
    new Experiments, given a certain objective. The abstraction
    follows closely the "scientific method". Agent is the entity
    that suggests new Experiments.

    Supporting entities are Analyzers and Finalizers. Framework
    is flexible enough to implement many sequential learning or
    optimization tasks, including active-learning, bayesian optimization
    or black-box optimization with local or global optima search.

    """
    def __init__(
        self,
        candidate_data,
        agent,
        experiment,
        analyzer,
        seed_data=None,
        create_seed=False,
        heuristic_stopper=np.inf,
        s3_prefix=None,
        s3_bucket=CAMD_S3_BUCKET,
        path=None,
        logger=None
    ):
        """
        Invokes a campaign from candidates, seed, agent, and other supporting
        entities for decision-making, experiment, and analysis.

        Args:
            candidate_data (pd.DataFrame): List of uids for candidate
                search space for active learning
            agent (HypothesisAgent): a subclass of HypothesisAgent
            experiment (Experiment): a subclass of Experiment
            analyzer (Analyzer): a subclass of Analyzer
            seed_data (pandas.DataFrame): Seed Data for active learning,
                index is to be the assumed uid
            create_seed (int): an initial seed size to create from the data
            heuristic_stopper (int): If int, the heuristic stopper will
                kick in to check if loop should be terminated after this
                many iterations, if no discoveries in past #n loops.
            s3_prefix (str): prefix which to prepend all s3 synced files with,
                if None is specified, s3 syncing will not occur
            s3_bucket (str): bucket name for s3 sync.  If not specified,
                CAMD will sync to the specified environment variable.
            path (str): local path in which to execute the loop, defaults
                to current folder if path is not provided
            logger (logging.Logger): logger for campaign progress
        """
        # Cloud parameters
        self.s3_prefix = s3_prefix
        self.s3_bucket = s3_bucket

        # Data parameters
        self.candidate_data = candidate_data
        self.seed_data = seed_data if seed_data is not None else pd.DataFrame()
        self.create_seed = create_seed
        self.history = pd.DataFrame()

        # Object parameters
        self.agent = agent
        self.experiment = experiment
        self.analyzer = analyzer

        # Other parameters
        # TODO: think about how to abstract this away from the loop
        self.heuristic_stopper = heuristic_stopper
        self.path = path if path else os.getcwd()
        os.chdir(self.path)

        # Internal data
        self._exp_raw_results = None

        # Check if there exists earlier iterations
        if os.path.exists(os.path.join(self.path, "iteration.json")):
            self.load("iteration")
            self.initialized = True
        else:
            self.iteration = 0
            self.initialized = False

        if self.initialized:
            self.create_seed = False
            self.load("job_status")
            self.experiment.job_status = self.job_status
            self.load("experiment", method="pickle")
            self.load("seed_data", method="pickle")
            self.load("consumed_candidates")
            self.load("loop_state", no_exist_fail=False)
            self.initialized = True
        else:
            self.submitted_experiment_requests = []
            self.consumed_candidates = []
            self.job_status = {}
            self.initialized = False
            self.loop_state = "UNSTARTED"

        if logger is None:
            self.logger = logging.Logger("camd")
            self.logger.addHandler(logging.StreamHandler())
        else:
            self.logger = logger

    def run(self, finalize=False):
        """
        This method applies a single iteration of the loop, and
        keeps record of everything in place.

        Each iteration consists of:
            1. Get results of requested experiments
            2. Load, Expand, Save seed_data
            3. Augment candidate_space
            4. Analyze results - Stop / Go
            5. Hypothesize
            6. Submit new experiments
        """
        if not self.initialized:
            raise ValueError("Campaign must be initialized.")

        # Get new results
        self.logger.info("{} {} state: Getting new results".format(self.type, self.iteration))
        self.experiment.monitor()
        new_experimental_results = self.experiment.get_results()
        os.chdir(self.path)

        # Load seed_data
        self.load("seed_data", method="pickle")

        # Analyze new results
        self.logger.info("{} {} state: Analyzing results".format(self.type, self.iteration))
        summary, new_seed_data = self.analyzer.analyze(
            new_experimental_results, self.seed_data
        )

        # Augment summary and seed
        self.history = self.history.append(summary)
        self.history = self.history.reset_index(drop=True)
        self.save("history", method="pickle")
        self.seed_data = new_seed_data
        self.save("seed_data", method="pickle")

        # Remove candidates from candidate space
        candidate_space = self.candidate_data.index.difference(
            self.consumed_candidates, sort=False
        ).tolist()
        self.candidate_data = self.candidate_data.loc[candidate_space]
        if len(self.candidate_data) == 0:
            self.logger.info("Candidate data exhausted.  Stopping loop.")
            return False

        # Campaign stopper if no discoveries in last few cycles.
        if self.iteration > self.heuristic_stopper:
            new_discoveries = self.history["new_discovery"][-3:].values.sum()
            if new_discoveries == 0:
                self.finalize()
                self.logger.info("Not enough new discoveries. Stopping the loop.")
                return False

        # Campaign stopper if finalization is desired but will be done
        # outside of run (e.g. auto_loop)
        if finalize:
            return False

        # Agent suggests new experiments
        self.logger.info(
            "{} {} state: Agent {} hypothesizing".format(
                self.type, self.iteration, self.agent.__class__.__name__
            )
        )
        suggested_experiments = self.agent.get_hypotheses(
            self.candidate_data, self.seed_data
        )

        # Campaign stopper if agent doesn't have anything to suggest.
        if len(suggested_experiments) == 0:
            self.finalize()
            self.logger.info("No agent suggestions. Stopping the loop.")
            return False

        # Experiments submitted
        self.logger.info("{} {} state: Running experiments".format(self.type, self.iteration))
        self.job_status = self.experiment.submit(suggested_experiments)
        self.save("job_status")

        self.save("experiment", method="pickle")

        self.consumed_candidates += suggested_experiments.index.values.tolist()
        self.save("consumed_candidates")

        self.iteration += 1
        self.save("iteration")
        return True

    def auto_loop(self, n_iterations=10, monitor=False,
                  initialize=False, save_iterations=False):
        """
        Runs the loop repeatedly, and locally. Contains
        option for backing up the loop in enumerated
        subdirectories for each iteration.

        Args:
            n_iterations (int): Number of iterations.
            monitor (bool): Use Experiment's monitor method to
                keep track of requested experiments.
            initialize (bool): whether to initialize the loop
                before starting
            save_iterations (bool): whether or not to save
                iterations in subdirectories of the working
                directory

        """
        if initialize:
            self.initialize()
            if save_iterations:
                self.loop_backup("-1")
        while n_iterations - self.iteration >= 0:
            self.logger.info("Iteration: {}".format(self.iteration))
            if not self.run():
                break
            self.logger.info("  Waiting for next round ...")
            if monitor:
                self.logger.info("Monitoring experiments")
                self.experiment.monitor()
            if save_iterations:
                self.loop_backup(str(self.iteration - 1))

        self.run(finalize=True)
        if monitor:
            self.logger.info("Monitoring experiments")
            self.experiment.monitor()
        self.finalize()

    def initialize(self, random_state=42):
        """
        Initializes a campaign. The primary goal of initialization is to ensure a proper seed exists. If create_seed
        is set in Campaign, it creates the seed by deploying the RandomAgent before the user-provided agent is
        deployed in the regular campaign iterations.
            random_state (int): ensures reproducible results.
        """
        if self.initialized:
            raise ValueError("Initialization may overwrite existing loop data. Exit.")
        if not self.seed_data.empty and not self.create_seed:
            self.logger.info(
                "{} {} state: Agent {} hypothesizing".format(
                    self.type, "initialization", self.agent.__class__.__name__
                )
            )
            suggested_experiments = self.agent.get_hypotheses(
                self.candidate_data, self.seed_data
            )
        elif self.create_seed:
            np.random.seed(seed=random_state)
            _agent = RandomAgent(self.candidate_data, n_query=self.create_seed)
            self.logger.info(
                "{} {} state: Agent {} hypothesizing".format(
                    self.type, "initialization", _agent.__class__.__name__
                )
            )
            suggested_experiments = _agent.get_hypotheses(self.candidate_data)
        else:
            raise ValueError(
                "No seed data available. Either supply or ask for creation."
            )
        self.analyzer._initial_seed_indices = self.seed_data.index.tolist()

        self.logger.info("{} {} state: Running experiments".format(self.type, self.iteration))
        self.job_status = self.experiment.submit(suggested_experiments)
        self.consumed_candidates = suggested_experiments.index.values.tolist()
        self.create_seed = False
        self.initialized = True

        self.save("job_status")
        self.save("seed_data", method="pickle")
        self.save("experiment", method="pickle")
        self.save("consumed_candidates")
        self.save("iteration")
        if self.s3_prefix:
            self.s3_sync()

    @property
    def type(self):
        """
        Convenience property for campaign type that
        gets the class name, mostly for logging

        Returns:
            (str): class name

        """
        return self.__class__.__name__

    def finalize(self):
        """
        Run finalization method for campaign
        if analyzer has finalize method

        Returns:
            None

        """
        self.logger.info("Finalizing campaign.")
        os.chdir(self.path)
        if hasattr(self.analyzer, "finalize"):
            self.analyzer.finalize(self.path)
        if self.s3_prefix:
            self.s3_sync()

    def load(self, data_holder, method="json", no_exist_fail=True):
        """
        Method to load stored object attributes

        Args:
            data_holder (str): attribute to be loaded
            method (str): method by which to load object,
                'pickle' and 'json' are currently supported
            no_exist_fail (bool): whether to throw error
                on non-existence of data

        Returns:
            None

        """
        if method == "pickle":
            m = pickle
            mode = "rb"
        elif method == "json":
            m = json
            mode = "r"
        else:
            raise ValueError("Unknown data save method")

        file_name = os.path.join(self.path, data_holder + "." + method)
        exists = os.path.exists(file_name)

        if exists:
            with open(file_name, mode) as f:
                self.__setattr__(data_holder, m.load(f))
        else:
            if no_exist_fail:
                raise IOError("No {} file exists".format(data_holder))
            else:
                self.__setattr__(data_holder, None)

    def save(self, data_holder, custom_name=None, method="json"):
        """
        Save method for storing campaign data

        Args:
            data_holder (str): attribute to be written to file
            custom_name (str): custom filename if desired
            method (str): method option for data storage,
                'json' or 'pickle' are supported

        Returns:
            None

        """
        if custom_name:
            _path = os.path.join(self.path, custom_name)
        else:
            _path = os.path.join(self.path, data_holder + "." + method)
        if method == "pickle":
            m = pickle
            mode = "wb"
        elif method == "json":
            m = json
            mode = "w"
        else:
            raise ValueError("Unknown data save method")
        with open(_path, mode) as f:
            m.dump(self.__getattribute__(data_holder), f)

        if self.s3_prefix:
            self.s3_sync()

    def s3_sync(self):
        """
        Syncs current run to s3_prefix and bucket
        """
        s3_sync(self.s3_bucket, self.s3_prefix, self.path)

    def loop_backup(self, target_dir):
        """
        Helper method to backup finished loop iterations.

        Args:
            target_dir (str, Path): directory to back up to

        Returns:
            (None)

        """
        self.logger.info("Backing loop up to s3")
        os.mkdir(os.path.join(self.path, target_dir))
        _files = os.listdir(self.path)
        for file_name in _files:
            full_file_name = os.path.join(self.path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(self.path, target_dir))

        if self.s3_prefix:
            self.s3_sync()
