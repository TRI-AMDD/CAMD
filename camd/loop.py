# Copyright Toyota Research Institute 2019
import os
import pickle
import json
import time
import numpy as np
import warnings

from monty.json import MSONable


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

        self.experiment = experiment(**experiment_params)
        self.experiment_params = experiment_params

        self.analyzer = analyzer(**analyzer_params)
        self.analyzer_params = analyzer_params

        self.seed_data = seed_data
        self.create_seed = create_seed

        # Check if there exists earlier iterations
        if os.path.exists(os.path.join(self.path, 'iteration.log')):
            with open(os.path.join(self.path, 'iteration.log'), 'r') as f:
                self.iteration = int(f.readline().rstrip('\n'))
        else:
            self.iteration = 0
        if self.iteration>0:
            self.create_seed=False
            warnings.warn("Turning off create_seed since there is existing iteration in path. Use clean folder to"
                          "to reinitialize from scratch.")

        self.stop = False
        self.submitted_experiment_requests = []
        self.job_status = {}

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


        if os.path.exists(os.path.join(self.path, 'iteration.log')):
            with open(os.path.join(self.path, 'iteration.log'), 'r') as f:
                self.iteration = int(f.readline().rstrip('\n'))

        print("1. Get new results")
        self.load('submitted_experiment_requests')

        if not self.job_status:
            self.load('job_status')
            self.experiment = self.experiment.from_job_status(self.experiment_params, self.job_status)

        new_experimental_results = self.experiment.get_results(self.submitted_experiment_requests)

        print("2. Load, expand, save seed_data")
        if self.iteration>0:
            self.load('seed_data')
            self.seed_data = self.seed_data.append(new_experimental_results)
        else:
            self.seed_data = new_experimental_results
        self.save('seed_data')

        print("3. Augment candidate space")
        self.candidate_space = list(set(self.candidate_data.index).difference(set(self.seed_data.index)))
        self.candidate_data = self.candidate_data.loc[self.candidate_space]

        print("4. Analyze results")
        self.results_new_uids, self.results_all_uids = self.analyzer.analyze(self.seed_data,
                                                                                        self.submitted_experiment_requests)
        self._discovered = np.array(self.submitted_experiment_requests)[self.results_new_uids].tolist()
        self.save('_discovered', custom_name='discovered_{}.pd'.format(self.iteration))

        print("5. Hypothesize")
        suggested_experiments = self.agent.get_hypotheses(self.candidate_data, self.seed_data)

        print("6. Submit new experiments")
        self.job_status = self.experiment.submit(suggested_experiments)
        self.save("job_status")

        self.submitted_experiment_requests = suggested_experiments
        self.save('submitted_experiment_requests')

        self.report()
        self.iteration+=1
        with open(os.path.join(self.path, 'iteration.log'), 'w') as f:
            f.write(str(self.iteration))

    def auto_loop(self, n_iterations=10, timeout=10, run_monitor=False):
        """
        Runs the loop repeatedly
        TODO: Stopping criterion from Analyzer
        Args:
            n_iterations (int): Number of iterations.
            timeout (int): Time (in seconds) to wait on idle for submitted experiments to finish.
            run_monitor (bool): Use Experiment's run_monitor method to keep track of requested experiments.
        """
        if self.create_seed:
            print("creating seed")
            self.initialize()
            time.sleep(timeout)
            print("finished creating seed")
        while n_iterations - self.iteration >= 0:
            print("Iteration: {}".format(self.iteration))
            self.run()
            print("  Waiting for next round ...")
            if run_monitor:
                self.experiment.run_monitor()
            time.sleep(timeout)

    def initialize(self, random_state=42):
        print("Initializing")
        np.random.seed(seed=random_state)
        suggested_experiments = np.random.choice(self.candidate_space, self.create_seed, replace=False).tolist()
        self.job_status = self.experiment.submit(suggested_experiments)
        self.save("job_status")
        self.submitted_experiment_requests = suggested_experiments
        self.save('submitted_experiment_requests')
        self.create_seed = False

    def report(self):
        with open(os.path.join(self.path, 'report.log'), 'a') as f:
            if self.iteration == 0:
                f.write("Iteration N_Discovery Total_Discovery N_query N_candidates model-CV\n")
            f.write("{:9} {:11} {:15} {:12} {:f}\n".format(self.iteration, np.sum(self.results_new_uids),
                                                           np.sum(self.results_all_uids), len(self.candidate_data),
                                                           self.agent.cv_score))

    def load(self, data_holder, method='pickle'):
        with open(os.path.join(self.path, data_holder+'.'+method), 'rb') as f:
            if method == 'pickle':
                m = pickle
            elif method == 'json':
                m = json
            else:
                raise ValueError("Unknown data save method")
            self.__setattr__(data_holder, m.load(f))

    def save(self, data_holder, custom_name=None, method='pickle'):
        if custom_name:
            _path = os.path.join(self.path, custom_name)
        else:
            _path = os.path.join(self.path, data_holder+'.'+method)
        with open(_path, 'wb') as f:
            if method == 'pickle':
                m = pickle
            elif method == 'json':
                m = json
            else:
                raise ValueError("Unknown data save method")
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