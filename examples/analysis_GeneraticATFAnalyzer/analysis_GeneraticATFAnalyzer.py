import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from camd.analysis import GenericATFAnalyzer

# settings and params
data_path = "."                                                       # where the results will be stored
fig_path = "."                                                     # where the generated figures are saved
num_seed = 10                                                            # size of seed dataset
threshold = -0.20                                                        # threshold defining effective collection
max_budget = 200                                                         # budget to run active learning
percentile = 0.01


def gather_seed_data_record(path, scenario, num_cycle):
    """
    Args:
        path (str):
            point to data folder
        scenario (str):
            common starting names for the log folder, e.g. "random-"
        num_cycle (int):
            number of cycles applied to such scenario, indicating 
            the number of log folders
    Returns:
        history_list (list):
            list of loaded history file
    """
    record_list = []
    for i in range(num_cycle):
        record = pickle.load(open(path+"/seed_data_%s_%s.pickle" % (scenario, str(i)), "rb"))
        record_list.append(record)
    return record_list


def run_analysis_alm(exploration_space_df, record_list):
    """
    run analysis one by one
    Args:
        exploration_space_df (pd.DataFrame):
            dataframe represent the whole exploration space
        record_target_df (list):
            list of pd.DataFrame, each pd.dataFrame is the seed_data_df loaded from 
            CAMD campaigin
            each dataframe with concatenated columns, each column are the target values
            selected by CAMD agent running campaign, length of the list should be
            seed_size + max_budget + 1
    Returns:
        deALM (np.array):
            results from GenericATFAnalyzer.gather_deALM()
        anyALM (np.array):
            results from GenericATFAnalyzer.gather_anyALM()
        allALM (np.array):
            results from GenericATFAnalyzer.gather_allALM()
        simALM (np.array):
            results from GenericATFAnalyzer.gather_simALM()
    """

    deALM = []
    anyALM = [] 
    allALM = [] 
    simALM = []
    for record in record_list:
        seed_data = pd.DataFrame(record[:num_seed])
        new_experimental_results = pd.DataFrame(record[num_seed:(num_seed+max_budget)])
        analyzer = GenericATFAnalyzer(exploration_space_df, percentile)
        summary, _ = analyzer.analyze(new_experimental_results, seed_data)
        deALM.append(summary['deALM'].to_list()[0])
        anyALM.append(summary['anyALM'].to_list()[0])
        allALM.append(summary['allALM'].to_list()[0])
        simALM.append(summary['simALM'].to_list()[0])
    return np.array(deALM), np.array(anyALM), np.array(allALM), np.array(simALM)


if __name__ == "__main__":
    # load data
    example_dataFrame_with_target_features = data_path+"/df.csv"
    exploration_space_df = pd.read_csv(example_dataFrame_with_target_features)
    # negate the target values because it was a maximizing campaign
    exploration_space_df['target'] = -exploration_space_df['target']
    # run analysis of sequential learning performance for random scenario
    record_list_random = gather_seed_data_record(data_path, scenario="random", num_cycle=3)
    # gather all target values and put into one df
    deALM_random, anyALM_random, allALM_random, simALM_random = \
        run_analysis_alm(exploration_space_df, record_list_random)
    # run analysis of sequential learning performance for ucb scenario
    record_list_ucb = gather_seed_data_record(data_path, scenario="ucb", num_cycle=3)
    # gather all target values and put into one df
    deALM_ucb, anyALM_ucb, allALM_ucb, simALM_ucb = \
        run_analysis_alm(exploration_space_df, record_list_ucb)

    # plot the deALM comparison
    # deALM
    fig, ax = plt.subplots(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_dealm_plot = plt.plot(np.arange(len(deALM_random[0])), deALM_random.mean(axis=0), '-or')
    ucb_dealm_plot = plt.plot(np.arange(len(deALM_ucb[0])), deALM_ucb.mean(axis=0), '-ob')
    ax.fill_between(np.arange(len(deALM_ucb[0])), deALM_ucb.mean(axis=0)-deALM_ucb.std(axis=0), 
                    deALM_ucb.mean(axis=0)+deALM_ucb.std(axis=0), color='blue', alpha=0.5)
    ax.fill_between(np.arange(len(deALM_random[0])), deALM_random.mean(axis=0)-deALM_random.std(axis=0), 
                    deALM_random.mean(axis=0)+deALM_random.std(axis=0), color='red', alpha=0.5)
    plt.axis([-1, max_budget+1, -1.5, 1.5])
    plt.legend([random_dealm_plot[0], ucb_dealm_plot[0]], ['Random Agent', 'UCB Agent'], 
               loc='upper right', frameon=False, fontsize=13)
    plt.plot(np.arange(len(deALM_random[0])), deALM_random.mean(axis=0), '-or')
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{de}$ALM', fontsize=15)
    plt.text(0, -1.0, "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/alm_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')

    # anyALM
    fig, ax = plt.subplots(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_anyalm_plot = plt.plot(np.arange(len(anyALM_random[0])), anyALM_random.mean(axis=0), '-or')
    ucb_anyalm_plot = plt.plot(np.arange(len(anyALM_ucb[0])), anyALM_ucb.mean(axis=0), '-ob')
    ax.fill_between(np.arange(len(anyALM_ucb[0])), anyALM_ucb.mean(axis=0)-anyALM_ucb.std(axis=0), 
                    anyALM_ucb.mean(axis=0)+anyALM_ucb.std(axis=0), color='blue', alpha=0.5)
    ax.fill_between(np.arange(len(anyALM_random[0])), anyALM_random.mean(axis=0)-anyALM_random.std(axis=0), 
                    anyALM_random.mean(axis=0)+anyALM_random.std(axis=0), color='red', alpha=0.5)
    plt.axis([-1, max_budget+1, -0.3, 1])
    plt.legend([random_anyalm_plot[0], ucb_anyalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{any}$ALM', fontsize=15)
    plt.text(0, 0.75, "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/any_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')

    # allALM
    fig, ax = plt.subplots(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_allalm_plot = plt.plot(np.arange(len(allALM_random[0])), allALM_random.mean(axis=0), '-or')
    ucb_allalm_plot = plt.plot(np.arange(len(allALM_ucb[0])), allALM_ucb.mean(axis=0), '-ob')
    ax.fill_between(np.arange(len(allALM_ucb[0])), allALM_ucb.mean(axis=0)-allALM_ucb.std(axis=0), 
                    allALM_ucb.mean(axis=0)+allALM_ucb.std(axis=0), color='blue', alpha=0.5)
    ax.fill_between(np.arange(len(allALM_random[0])), allALM_random.mean(axis=0)-allALM_random.std(axis=0), 
                    allALM_random.mean(axis=0)+allALM_random.std(axis=0), color='red', alpha=0.5)
    plt.axis([-1, max_budget+1, -0.1, 1])
    plt.legend([random_allalm_plot[0], ucb_allalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{all}$ALM', fontsize=15)
    plt.text(0, 0.15, "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/all_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')

    # simALM
    fig, ax = plt.subplots(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)
    random_simalm_plot = plt.plot(np.arange(len(simALM_random[0])), simALM_random.mean(axis=0), '-or')
    ucb_simalm_plot = plt.plot(np.arange(len(simALM_ucb[0])), simALM_ucb.mean(axis=0), '-ob')
    ax.fill_between(np.arange(len(simALM_ucb[0])), simALM_ucb.mean(axis=0)-simALM_ucb.std(axis=0), 
                    simALM_ucb.mean(axis=0)+simALM_ucb.std(axis=0), color='blue', alpha=0.5)
    ax.fill_between(np.arange(len(simALM_random[0])), simALM_random.mean(axis=0)-simALM_random.std(axis=0), 
                    simALM_random.mean(axis=0)+simALM_random.std(axis=0), color='red', alpha=0.5)
    plt.xlim([-1, max_budget+1])
    plt.legend([random_simalm_plot[0], ucb_simalm_plot[0]], ['Random Agent', 'UCB Agent'], frameon=False, fontsize=13)
    plt.xlabel('Number of Iteration', fontsize=15)
    plt.ylabel('$^{sim}$ALM', fontsize=15)
    plt.text(0, min(simALM_ucb.mean(axis=0).min(), simALM_random.mean(axis=0).min()), 
             "Number of Seed: %s\n" % str(num_seed), dict(size=14))
    # plt.show()
    plt.savefig("%s/sim_metric_alm_numSeed_%s_maxBudget_%s.png" % (fig_path, str(num_seed), str(max_budget)), 
                dpi=100, bbox_inches='tight')
