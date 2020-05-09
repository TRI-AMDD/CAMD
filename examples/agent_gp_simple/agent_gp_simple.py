# Copyright Toyota Research Institute 2019
from camd.campaigns.base import Campaign

from camd.agent.stability import GaussianProcessStabilityAgent
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler
from camd.utils.data import load_default_atf_data

##########################################################
# Load dataset and filter by N_species of 2 or less
##########################################################
df = load_default_atf_data()

##########################################################
# Binary stable material discovery GP based agent recipe
##########################################################
n_seed = 5000  # Starting sample size
n_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = GaussianProcessStabilityAgent(n_query=n_query,
    hull_distance= 0.05,
    alpha=0.5
    )
analyzer = StabilityAnalyzer(hull_distance=0.05)
experiment = ATFSampler(dataframe=df)
candidate_data = df
##########################################################
new_loop = Campaign(candidate_data, agent, experiment, analyzer, create_seed=n_seed)

new_loop.auto_loop(n_iterations=4, initialize=True)
