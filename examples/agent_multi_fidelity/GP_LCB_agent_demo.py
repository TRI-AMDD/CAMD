"""
This script demonstrate how to run a multi-fidelity campaign using epsilon-greedy agent. The campaign was run using bounday condition acquisition (where all the DFT data is available). The example dataset used in this demo is the bandgap dataset in this folder. Please see "generate_example_featurized_dataset" notebook on how we generate/featurized the dataset.
"""

import os
import pandas as pd
from monty.os import cd
from camd.agent.multi_fidelity import GPMultiAgent
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
# os.system('rm -rf Results/GP') # If the folder already exist, delete it first
os.system('mkdir -p Results/GP')

# Provide inputs for the campaign
N_query = 20
iterations = 5
data_features = [column for column in df if column.startswith("MagpieData")]

####################################################
# Run the campaign
agent = GPMultiAgent(candidate_data=candidate_data, seed_data=seed_data,
                     features=data_features, target_prop='bandgap', target_prop_val=1.8, 
                     total_budget=N_query, alpha=0.1)
experiment = ATFSampler(dataframe=candidate_data)
analyzer = MultiAnalyzer(target_prop='bandgap', prop_range=[1.6, 2.0])

with cd('Results/GP'):
    campaign = Campaign(candidate_data=candidate_data, seed_data=seed_data,
                        agent=agent, experiment=experiment, analyzer=analyzer)
    campaign.auto_loop(n_iterations=iterations, initialize=True)