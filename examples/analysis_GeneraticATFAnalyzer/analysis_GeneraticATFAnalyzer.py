import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from camd.analysis import GenericATFAnalyzer

# settings and params
cal_path = "camd_campaigns"                                              # where the results will be stored
fig_path = "figures"                                                     # where the generated figures are saved
num_seed = 10                                                            # size of seed dataset
threshold = -0.20                                                        # threshold defining effective collection
max_budget = 200                                                         # budget to run active learning


def gather_seed_data_record(path, scenario, num_cycle):
    """
        Args:
            - path (str):
                point to where CAMD campaign happens
            - scenario (str):
                common starting names for the log folder, e.g. "random-"
            - num_cycle (int):
                number of cycles applied to such scenario, indicating 
                the number of log folders
        Returns:
            - history_list (list):
                list of loaded history file
    """
    path = os.path.join(path, scenario)
    record_list = []
    for i in range(num_cycle):
        record = pickle.load(open(path+'-'+str(i)+'/seed_data.pickle', "rb"))
        record_list.append(record)
    return record_list


def run_analysis_alm(exploration_space_df, record_list):
    """
        run analysis one by one
        Args:
            - exploration_space_df (pd.DataFrame):
                dataframe represent the whole exploration space
            - record_target_df (list):
                list of pd.DataFrame, each pd.dataFrame is the seed_data_df loaded from 
                CAMD campaigin
                each dataframe with concatenated columns, each column are the target values
                selected by CAMD agent running campaign, length of the list should be
                    seed_size + max_budget + 1
    """
    almAnalyzer = GenericATFAnalyzer(threshold=threshold, seed_size=num_seed)
    deALM_val = almAnalyzer.gather_deALM(exploration_space_df, record_list)
    anyALM_val = almAnalyzer.gather_anyALM(exploration_space_df, record_list, percentile=0.01)
    allALM_val = almAnalyzer.gather_allALM(exploration_space_df, record_list, percentile=0.1)
    simALM_val = almAnalyzer.gather_simALM(record_list)
    return deALM_val, anyALM_val, allALM_val, simALM_val


if __name__ == "__main__":
    # settings
    camd_campaign_path = cal_path
    example_dataFrame_with_target_features = "data/sampleID_target_df_3mA_all_features.csv"
    exploration_space_df = pd.read_csv(example_dataFrame_with_target_features)
    # negate the target values because it was a maximizing campaign
    exploration_space_df['target'] = -exploration_space_df['target']
    # run analysis of sequential learning performance for random scenario
    record_list_random = gather_seed_data_record(cal_path, scenario="random", num_cycle=5)
    # gather all target values and put into one df
    deALM_random, anyALM_random, allALM_random, simALM_random = \
        run_analysis_alm(exploration_space_df, record_list_random)
    # run analysis of sequential learning performance for ucb scenario
    record_list_ucb = gather_seed_data_record(cal_path, scenario="ucb-s1-1", num_cycle=5)
    # gather all target values and put into one df
    # record_campaign_target_df_ucb = pd.concat([x['target'].reset_index(drop=True) for x in record_list_ucb], axis=1)
    deALM_ucb, anyALM_ucb, allALM_ucb, simALM_ucb = run_analysis_alm(exploration_space_df, record_list_ucb)

    # plot the deALM comparison
    # deALM
    fig = plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_dealm_plot = plt.plot(np.arange(len(deALM_random[0])), deALM_random.mean(axis=0), '-or')
    ucb_dealm_plot = plt.plot(np.arange(len(deALM_ucb[0])), deALM_ucb.mean(axis=0), '-ob')
    plt.axis([-1, max_budget+1, -1, 1])
    plt.legend([random_dealm_plot[0], ucb_dealm_plot[0]], ['Random Agent', 'UCB Agent'],
               loc='upper right', frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{de}$ALM', fontsize=15)
    plt.text(0, -0.85, "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/alm_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)),
                dpi=100, bbox_inches='tight')

    # anyALM
    fig = plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_anyalm_plot = plt.plot(np.arange(len(anyALM_random[0])), anyALM_random.mean(axis=0), '-or')
    ucb_anyalm_plot = plt.plot(np.arange(len(anyALM_ucb[0])), anyALM_ucb.mean(axis=0), '-ob')
    plt.axis([-1, max_budget+1, -0.1, 1])
    plt.legend([random_anyalm_plot[0], ucb_anyalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{any}$ALM', fontsize=15)
    plt.text(100, 0.15, "Number of Seed: %s\n" % str(num_seed), dict(size=14))        
    # plt.show()
    plt.savefig("%s/any_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')

    # allALM
    fig = plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)

    random_allalm_plot = plt.plot(np.arange(len(allALM_random[0])), allALM_random.mean(axis=0), '-or')
    ucb_allalm_plot = plt.plot(np.arange(len(allALM_ucb[0])), allALM_ucb.mean(axis=0), '-ob')
    plt.axis([-1, max_budget+1, -0.1, 1])
    plt.legend([random_allalm_plot[0], ucb_allalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{all}$ALM', fontsize=15)
    plt.text(0, 0.15, "Number of Seed: %s\n" % str(num_seed), dict(size=14))        
    # plt.show()
    plt.savefig("%s/all_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')

    # simALM
    fig = plt.figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_simalm_plot = plt.plot(np.arange(len(simALM_random[0])), simALM_random.mean(axis=0), '-or')
    ucb_simalm_plot = plt.plot(np.arange(len(simALM_ucb[0])), simALM_ucb.mean(axis=0), '-ob')
    plt.xlim([-1, max_budget+1])
    plt.legend([random_simalm_plot[0], ucb_simalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{sim}$ALM', fontsize=15)
    plt.text(0, 0.15, "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/sim_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')
