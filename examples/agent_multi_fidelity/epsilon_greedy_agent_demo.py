"""
This script demonstrates how to run a multi-fidelity campaign using epsilon-greedy agent. The campaign was run using bounday condition acquisition (where all the DFT data is available in the seed data). The example dataset used in this demo is the bandgap dataset (located in the current folder). Please see "generate_example_featurized_dataset" notebook on how we generated/featurized the dataset.
"""

import os
import pandas as pd
from monty.os import cd
from sklearn.svm import SVR
from tqdm import tqdm
from camd.agent.base import RandomAgent
from camd.agent.multi_fidelity import EpsilonGreedyMultiAgent
from camd.analysis import MultiAnalyzer
from camd.experiment.base import ATFSampler
from camd.campaigns.base import Campaign

####################################################
# Load dataset and determine seed and candidate data
df = pd.read_csv("example_dataset.csv", index_col=0)
seed_data = df.loc[df.expt_data==0]
candidate_data = df.loc[df.expt_data==1]

####################################################
# Create the folder where the results go
# os.system('rm -rf Results/{folder name}') # If a result folder already exist, delete it first
os.system('mkdir -p Results/epsilon_greedy_2Fi')

# Provide inputs for the campaign
N_query = 20
iterations = 10
data_features = [column for column in df if column.startswith("MagpieData")]
model = SVR(C=10)

####################################################
# Run campaigns
####################################################
# multi-fidelity campaugn using SVR
agent = EpsilonGreedyMultiAgent(candidate_data=candidate_data, seed_data=seed_data,
                               features=data_features, target_prop='bandgap', target_prop_val=1.8,
                                model=model, total_budget=N_query, highFi_query_frac=1)
experiment = ATFSampler(dataframe=candidate_data)
analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])

with cd('Results/epsilon_greedy_2Fi'):
    campaign = Campaign(candidate_data=candidate_data, seed_data=seed_data,
                        agent=agent, experiment=experiment, analyzer=analyzer)
    campaign.auto_loop(n_iterations=iterations, initialize=True)
    

################################################################
# if you also want to compare the multi-fidelity agent to random 
# and single fidelity agent. Uncomment the code below
################################################################
# # Random campaign
# os.system('mkdir -p Results/random')
# agent = RandomAgent(n_query = N_query)
# experiment = ATFSampler(dataframe=candidate_data)
# analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])
# with cd('Results/random'):
#     campaign = Campaign(candidate_data=candidate_data, seed_data=seed_data, 
#                         agent=agent, experiment=experiment, analyzer=analyzer)
#     campaign.auto_loop(n_iterations=iterations, initialize=True)

# # single fidelity campaign using SVR
# os.system('mkdir -p Results/epsilon_greedy_1Fi')
# agent = EpsilonGreedyMultiAgent(candidate_data=candidate_data, seed_data=None,
#                                features=data_features, target_prop='bandgap', target_prop_val=1.8,
#                                 model=model, total_budget=N_query, highFi_query_frac=1)
# experiment = ATFSampler(dataframe=candidate_data)
# analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])

# with cd('Results/epsilon_greedy_1Fi'):
#     campaign = Campaign(candidate_data=candidate_data, seed_data=None, create_seed=N_query,
#                         agent=agent, experiment=experiment, analyzer=analyzer)
#     campaign.auto_loop(n_iterations=iterations, initialize=True)

