# Copyright Toyota Research Institute 2019
from camd.campaigns.loop import Campaign

from camd.agent.stability import BaggedGaussianProcessStabilityAgent
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler
from camd.utils.data import load_default_atf_data

##########################################################
# Load dataset and filter by N_species of 2 or less
##########################################################
df = load_default_atf_data()

##########################################################
# Binary stable material discovery GP bagging
##########################################################
n_seed = 5000  # Starting sample size
n_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = BaggedGaussianProcessStabilityAgent
agent_params = {
    'n_query': n_query,
    'hull_distance': 0.05,  # Distance to hull to consider a finding as discovery (eV/atom)
    'alpha': 0.5,
    'n_estimators': 5,
    'max_samples': 1000
    }

analyzer = StabilityAnalyzer
analyzer_params = {'hull_distance': 0.05}
experiment = ATFSampler
experiment_params = {'dataframe': df}
candidate_data = df
##########################################################
new_loop = Campaign(candidate_data, agent, experiment, analyzer,
                    agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params,
                    create_seed=n_seed)

new_loop.auto_loop(n_iterations=4, timeout=5, initialize=True)
