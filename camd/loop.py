# Copyright Toyota Research Institute 2019

import abc

import os
import pickle
import json
import numpy as np
from monty.json import MSONable

from camd.experiment import get_dft_calcs_aft


class ActiveLearningLoop(MSONable):
    def __init__(self, path, data, agent_class, experiment_class, analyzer_class=None):
        """

        Args:
            path (str or Path-like): path in which to execute the loop
            data (dict): starting data
            agent (HypothesisAgent): a subclass of HypothesisAgent
            experiment (Experiment): a subclass of Experiment
            analyzer (Analyzer): a subclass of Analyzer, in principle this
            is optional
        """
        self.path = path
        self.data = data
        self.agent = agent_class(data)
        self.experiment_class = experiment_class
        self.analyzer_class = analyzer_class

    def construct(self, agent_params):
        pass

# TODO: subsampling capability should be a functionality of hypo
#  generation.  df_sub here should just be repalced with an
#  option to downselect to e.g. chemistry


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
    suggested_experiments = agent.hypotheses()
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
