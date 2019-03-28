# Copyright Toyota Research Institute 2019

import os
import pickle
import json
import numpy as np

from camd.analysis import get_stabilities_from_data
from camd.experiment import get_dft_calcs_aft


def aft_loop(path, df, df_sub, N_seed, N_query, hull_distance, agent,  agent_params):
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
    true_stabilities, ss, ns = get_stabilities_from_data(seed_data, new_experiment_requests, hull_distance)
    sa = np.array([true_stabilities[uid] for uid in new_experiment_requests]) <= hull_distance
    nd = np.sum(sa)

    # Learn and Hypothesize
    candidate_space = list(set(df_sub.index).difference(set(seed_data.index)))
    candidate_data = get_features_aft(candidate_space, df)

    print "Agent working"
    agent = agent(candidate_data, seed_data, hull_distance, N_query, **agent_params)

    suggested_experiments = agent.hypotheses()

    with open(os.path.join(path, 'next_experiments_requests.json'), 'w') as f:
        json.dump(suggested_experiments, f)

    with open(os.path.join(path, 'discovered_{}.json'.format(iteration)), 'w') as f:
        json.dump(np.array(new_experiment_requests)[sa].tolist(), f)

    with open(os.path.join(path, 'discovery.log'), 'a') as f:
        if iteration == 0:
            f.write("Iteration Discovery N_query N_candidates N_stable model-CV\n")
        f.write("{:9} {:9} {:7} {:11} {:8} {:f}\n".format(iteration, nd, N_query, len(candidate_space), ns,
                                                          agent.cv_score))



def get_features_aft(ids, d):
    """
    Placeholder function that mocks featurization of structures
    """
    return d.loc[ids]



