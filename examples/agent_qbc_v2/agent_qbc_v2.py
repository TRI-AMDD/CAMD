# Copyright Toyota Research Institute 2019
from camd.campaigns.base import Campaign

from sklearn.neural_network import MLPRegressor
from camd.agent.stability import QBCStabilityAgent
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler
from camd.utils.data import load_default_atf_data

##########################################################
# Load dataset and filter by n_species of 2 or less
##########################################################
df = load_default_atf_data()

##########################################################
# Binary stable material discovery QBC based agent recipe
##########################################################
n_seed = 5000  # Starting sample size - a seed of this size will be randomly chosen.
n_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = QBCStabilityAgent(MLPRegressor(hidden_layer_sizes=(84, 50)),
                          n_query=n_query,
                          hull_distance=0.05,
                          training_fraction=0.5,
                          n_members=5
                          )
analyzer = StabilityAnalyzer(hull_distance=0.05)
experiment = ATFSampler(dataframe=df)
candidate_data = df

new_loop = Campaign(candidate_data, agent, experiment, analyzer, create_seed=n_seed)
new_loop.auto_loop(n_iterations=4, initialize=True)
