# Copyright Toyota Research Institute 2019

import os
import pickle
import json
import numpy as np

from camd.analysis import AnalyzeStability
from camd.experiment import get_dft_calcs_aft


def aft_loop(path, df, df_sub, N_seed, N_query, agent, agent_params, analyzer, analyzer_params):
    """
    After-the-fact execution iterator for stable material discovery.
    :param path:
    :param df:
    :param df_sub:
    :param N_seed:
    :param N_query:
    :param hull_distance:
    :param agent:
    :param agent_params:
    :param analyzer:
    :param analyzer_params:
    :return:
    """

    iteration = 0
    if os.path.exists(os.path.join(path, 'iteration.log')):
        with open(os.path.join(path, 'iteration.log'), 'r') as f:
            iteration = int(f.readline().rstrip('\n'))
    with open(os.path.join(path, 'iteration.log'), 'w') as f:
        f.write(str(iteration + 1))

    if iteration == 0:
        seed_data = df.sample(n=N_seed)
        with open(os.path.join(path, 'seed_data.pd'), 'w') as f:
            pickle.dump(seed_data, f)
        new_experiment_requests = []
    else:
        with open(os.path.join(path, 'seed_data.pd'), 'r') as f:
            seed_data = pickle.load(f)
        with open(os.path.join(path, 'next_experiments_requests.json'), 'r') as f:
            new_experiment_requests = json.load(f)

            print "Experiment"
            new_experimental_results = get_dft_calcs_aft(new_experiment_requests, df)

            seed_data = seed_data.append(new_experimental_results)
        with open(os.path.join(path, 'seed_data.pd'), 'w') as f:
            pickle.dump(seed_data, f)

    print "Analysis"
    analyzer = analyzer(seed_data, new_experiment_requests, **analyzer_params)
    results_new_uids, results_all_uids = analyzer.analysis()
    with open(os.path.join(path, 'discovered_{}.json'.format(iteration)), 'w') as f:
        json.dump(np.array(new_experiment_requests)[results_new_uids].tolist(), f)

    # Learn
    candidate_space = list(set(df_sub.index).difference(set(seed_data.index)))
    candidate_data = get_features_aft(candidate_space, df)

    # Hypothesize
    print "Hypothesize: Agent working"
    agent = agent(candidate_data, seed_data, N_query, **agent_params)
    suggested_experiments = agent.hypotheses()
    with open(os.path.join(path, 'next_experiments_requests.json'), 'w') as f:
        json.dump(suggested_experiments, f)

    # Report out
    with open(os.path.join(path, 'discovery.log'), 'a') as f:
        if iteration == 0:
            f.write("Iteration N_Discovery Total_Discovery N_query N_candidates model-CV\n")
        f.write("{:9} {:11} {:15} {:7} {:12} {:f}\n".format(iteration, np.sum(results_new_uids),
                                                            np.sum(results_all_uids), N_query, len(candidate_space),
                                                            agent.cv_score))


def get_features_aft(ids, d):
    """
    Placeholder function that mocks featurization of structures
    """
    return d.loc[ids]



