# Copyright Toyota Research Institute 2019
import os
import pickle
import json
import time
import numpy as np
import warnings
import pandas as pd

from monty.json import MSONable
from camd.utils.s3 import cache_s3_objs
from camd import S3_CACHE
from camd.agent.base import RandomAgent

# TODO:
#  - improve the stopping scheme

class Loop(MSONable):
    def __init__(self, path, candidate_data, agent, experiment, analyzer,
                 agent_params=None, experiment_params=None, analyzer_params=None, seed_data=None, create_seed=False):
        """

        Args:
            path (str): path in which to execute the loop
            candidate_data (pd.DataFrame): List of uids for candidate search space for active learning
            agent (HypothesisAgent): a subclass of HypothesisAgent
            experiment (Experiment): a subclass of Experiment
            analyzer (Analyzer): a subclass of Analyzer
            seed_data (pandas.DataFrame): Seed Data for active learning, index is to be the assumed uid
        """
        self.path = path

        self.candidate_data = candidate_data
        self.candidate_space = list(candidate_data.index)

        self.agent = agent(**agent_params)
        self.agent_params = agent_params

        self.experiment = experiment(experiment_params)
        self.experiment_params = experiment_params

        self.analyzer = analyzer(**analyzer_params)
        self.analyzer_params = analyzer_params

        self.seed_data = seed_data if seed_data is not None else pd.DataFrame()
        self.create_seed = create_seed

        # Check if there exists earlier iterations
        if os.path.exists(os.path.join(self.path, 'iteration.json')):
            self.load('iteration')
            self.initialized = True
        else:
            self.iteration = 0
            self.initialized = False

        if self.initialized:
            self.create_seed = False
            self.load('job_status')
            self.experiment = self.experiment.from_job_status(self.experiment_params, self.job_status)
            self.load('submitted_experiment_requests')
            self.load('seed_data', method='pickle')
            self.initialized = True
        else:
            self.submitted_experiment_requests = []
            self.job_status = {}
            self.initialized = False

    def run(self):
        """
        This method applies a single iteration of the active-learning loop, and keeps record of everything in place.

        Each iteration consists of:
            1. Get results of requested experiments
            2. Load, Expand, Save seed_data
            3. Augment candidate_space
            4. Analyze results - Stop / Go
            5. Hypothesize
            6. Submit new experiments
        """
        if not self.initialized:
            raise ValueError("Loop needs to be properly initialized.")

        # Get new results
        print("Loop {} state: Getting new results".format(self.iteration))
        self.load('submitted_experiment_requests')
        new_experimental_results = self.experiment.get_results(self.submitted_experiment_requests)

        # Load, expand, save seed_data
        self.load('seed_data', method='pickle')
        self.seed_data = self.seed_data.append(new_experimental_results)
        self.save('seed_data', method='pickle')

        # Augment candidate space
        self.candidate_space = list(set(self.candidate_data.index).difference(set(self.seed_data.index)))
        self.candidate_data = self.candidate_data.loc[self.candidate_space]

        # Analyze results
        print("Loop {} state: Analyzing results".format(self.iteration))
        self.results_new_uids, self.results_all_uids = self.analyzer.analyze(self.seed_data,
                                                                            self.submitted_experiment_requests)
        self._discovered = np.array(self.submitted_experiment_requests)[self.results_new_uids].tolist()
        self.save('_discovered', custom_name='discovered_{}.json'.format(self.iteration))

        # Agent suggests new experiments
        print("Loop {} state: Agent {} hypothesizing".format(self.iteration, self.agent.__class__.__name__))
        suggested_experiments = self.agent.get_hypotheses(self.candidate_data, self.seed_data)

        # Stop-gap loop stopper.
        if len(suggested_experiments) == 0:
            raise ValueError("No space left to explore. Stopping the loop.")

        # Experiments submitted
        print("Loop {} state: Running experiments".format(self.iteration))
        self.job_status = self.experiment.submit(suggested_experiments)
        self.save("job_status")

        self.submitted_experiment_requests = suggested_experiments
        self.save('submitted_experiment_requests')

        self.report()
        self.iteration+=1
        self.save("iteration")

    def auto_loop(self, n_iterations=10, timeout=10, monitor=False):
        """
        Runs the loop repeatedly
        TODO: Stopping criterion from Analyzer
        Args:
            n_iterations (int): Number of iterations.
            timeout (int): Time (in seconds) to wait on idle for submitted experiments to finish.
            monitor (bool): Use Experiment's monitor method to keep track of requested experiments.
        """
        while n_iterations - self.iteration >= 0:
            print("Iteration: {}".format(self.iteration))
            self.run()
            print("  Waiting for next round ...")
            if monitor:
                self.experiment.monitor()
            time.sleep(timeout)

    def initialize(self, random_state=42):
        if self.initialized:
            raise ValueError("Initialization may overwrite existing loop data. Exit.")
        if not self.seed_data.empty and not self.create_seed:
            print("Loop {} state: Agent {} hypothesizing".format('initialization', self.agent.__class__.__name__))
            suggested_experiments = self.agent.get_hypotheses(self.candidate_data, self.seed_data)
        elif self.create_seed:
            np.random.seed(seed=random_state)
            _agent = RandomAgent(self.candidate_data, N_query=self.create_seed)
            print("Loop {} state: Agent {} hypothesizing".format('initialization', _agent.__class__.__name__))
            suggested_experiments = _agent.get_hypotheses(self.candidate_data)
        else:
            raise ValueError("No seed data available. Either supply or ask for creation.")

        print("Loop {} state: Running experiments".format(self.iteration))
        self.job_status = self.experiment.submit(suggested_experiments)
        self.submitted_experiment_requests = suggested_experiments
        self.create_seed = False
        self.initialized = True

        self.save("job_status")
        self.save("seed_data", method='pickle')
        self.save('submitted_experiment_requests')
        self.save("iteration")

    def initialize_with_icsd_seed(self, random_state=42):
        cache_s3_objs(["camd/shared-data/oqmd1.2_icsd_featurized_clean_v2.pickle"])
        self.seed_data = pd.read_pickle(os.path.join(S3_CACHE,
                                                     "camd/shared-data/oqmd1.2_icsd_featurized_clean_v2.pickle"))
        self.initialize(random_state=random_state)

    def report(self):
        with open(os.path.join(self.path, 'report.log'), 'a') as f:
            if self.iteration == 0:
                f.write("Iteration N_Discovery Total_Discovery N_candidates model-CV\n")
            report_string = "{:9} {:11} {:15} {:12} {:f}\n".format(self.iteration, np.sum(self.results_new_uids),
                                                           np.sum(self.results_all_uids), len(self.candidate_data),
                                                           self.agent.cv_score)
            f.write(report_string)

    def load(self, data_holder, method='json'):
        if method == 'pickle':
            m = pickle
            mode = 'rb'
        elif method == 'json':
            m = json
            mode = 'r'
        else:
            raise ValueError("Unknown data save method")
        with open(os.path.join(self.path, data_holder+'.'+method), mode) as f:

            self.__setattr__(data_holder, m.load(f))

    def save(self, data_holder, custom_name=None, method='json'):
        if custom_name:
            _path = os.path.join(self.path, custom_name)
        else:
            _path = os.path.join(self.path, data_holder+'.'+method)
        if method == 'pickle':
            m = pickle
            mode = 'wb'
        elif method == 'json':
            m = json
            mode = 'w'
        else:
            raise ValueError("Unknown data save method")
        with open(_path, mode) as f:
            m.dump(self.__getattribute__(data_holder), f)

    def get_state(self):
        pass

# a temporary helper function that first creates a domain and sets up a Loop
def get_structure_campaign(domain_params, loop_params):
    from camd.domain import StructureDomain
    domain = StructureDomain(domain_params)
    candidates = domain.candidates()
    return Loop(candidate_data=candidates, **loop_params)

# a temporary helper function that first creates a domain and sets up a Loop
def get_structure_campaign_from_bounds(bounds, domain_params, loop_params):
    from camd.domain import StructureDomain
    domain = StructureDomain.from_bounds(bounds, **domain_params)
    candidates = domain.candidates()
    return Loop(candidate_data=candidates, **loop_params)