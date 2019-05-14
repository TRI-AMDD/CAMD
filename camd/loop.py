# Copyright Toyota Research Institute 2019

import abc

import os
import pickle
import json
import time
import numpy as np
import warnings

from monty.json import MSONable

from camd.experiment.dft import get_dft_calcs_aft


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
        self.experiment.submit(suggested_experiments)
        self.submitted_experiment_requests = suggested_experiments
        self.save('submitted_experiment_requests')

        self.report()
        self.iteration+=1
        with open(os.path.join(self.path, 'iteration.log'), 'w') as f:
            f.write(str(self.iteration))

    def auto_loop(self, wait_time=10):
        """
        Runs the loop repeatedly
        TODO: Stopping criterion from Analyzer
        Args:
            wait_time (int): Time (in seconds) to wait on idle for submitted experiments to finish.
        """
        if self.create_seed:
            print("creating seed")
            self.initialize()
            time.sleep(wait_time)
            print("finished creating seed")
        while True:
            print("Iteration: {}".format(self.iteration))
            self.run()
            print("  Waiting for next round ...")
            time.sleep(wait_time)

    def initialize(self, random_state=42):
        print("Initializing")
        np.random.seed(seed=random_state)
        suggested_experiments = np.random.choice(self.candidate_space, self.create_seed, replace=False).tolist()
        self.experiment.submit(suggested_experiments)
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

    def load(self, data_holder):
        with open(os.path.join(self.path, data_holder+'.pd'), 'rb') as f:
            self.__setattr__(data_holder, pickle.load(f))

    def save(self, data_holder, custom_name=None):
        if custom_name:
            _path = os.path.join(self.path, custom_name)
        else:
            _path = os.path.join(self.path, data_holder+'.pd')
        with open(_path, 'wb') as f:
            pickle.dump(self.__getattribute__(data_holder), f)



def aft_loop(path, df, df_sub, n_seed, n_query, agent,
             agent_params, analyzer, analyzer_params):
    """
    After-the-fact execution iterator for stable material discovery.

    Args:
        path (str, Path): path for running the loop, i. e. where
            the output files will be output and the input files
            will be read from
        df (DataFrame): initial data set
        df_sub (DataFrame):
        n_seed (int): starting sample size
        n_query (int): starting query
        agent (HypothesisAgent):
        agent_params (dict): kwarg dict for the invocation
            of the agent
        analyzer (Analyzer): analyzer for postprocessing
            analysis
        analyzer_params (dict): kwarg dict parameters for
            invocation of the Analyzer

    Returns:
        (None): none, iterates on loop

    """

    # Get current iteration from log file
    iteration = 0
    if os.path.exists(os.path.join(path, 'iteration.log')):
        with open(os.path.join(path, 'iteration.log'), 'r') as f:
            iteration = int(f.readline().rstrip('\n'))
    with open(os.path.join(path, 'iteration.log'), 'w') as f:
        f.write(str(iteration + 1))

    # Generate seed data if on first iteration
    if iteration == 0:
        seed_data = df.sample(n=n_seed)
        with open(os.path.join(path, 'seed_data.pd'), 'wb') as f:
            pickle.dump(seed_data, f)
        new_experiment_requests = []
    else:
        with open(os.path.join(path, 'seed_data.pd'), 'rb') as f:
            seed_data = pickle.load(f)
        with open(os.path.join(path, 'next_experiments_requests.json'), 'r') as f:
            new_experiment_requests = json.load(f)

            # Run new experiments
            print("Experiment")
            new_experimental_results = get_dft_calcs_aft(
                new_experiment_requests, df)

            seed_data = seed_data.append(new_experimental_results)
        with open(os.path.join(path, 'seed_data.pd'), 'wb') as f:
            pickle.dump(seed_data, f)

    # Run analysis
    print("Analysis")
    analyzer = analyzer(seed_data, new_experiment_requests, **analyzer_params)
    results_new_uids, results_all_uids = analyzer.analysis()
    with open(os.path.join(path, 'discovered_{}.json'.format(iteration)), 'w') as f:
        json.dump(np.array(new_experiment_requests)[results_new_uids].tolist(), f)

    # Learn
    candidate_space = list(set(df_sub.index).difference(set(seed_data.index)))
    candidate_data = get_features_aft(candidate_space, df)

    # Hypothesize new experiments
    print("Hypothesize: Agent working")
    agent = agent(candidate_data, seed_data, n_query, **agent_params)
    suggested_experiments = agent.get_hypotheses()
    with open(os.path.join(path, 'next_experiments_requests.json'), 'w') as f:
        json.dump(suggested_experiments, f)

    # Report out
    with open(os.path.join(path, 'discovery.log'), 'a') as f:
        if iteration == 0:
            f.write("Iteration N_Discovery Total_Discovery N_query N_candidates model-CV\n")
        f.write("{:9} {:11} {:15} {:7} {:12} {:f}\n".format(iteration, np.sum(results_new_uids),
                                                            np.sum(results_all_uids), n_query, len(candidate_space),
                                                            agent.cv_score))


def get_features_aft(ids, d):
    """
    Placeholder function that mocks featurization of structures
    """
    return d.loc[ids]
