# Copyright Toyota Research Institute 2019
from camd.loop import Loop

from sklearn.neural_network import MLPRegressor
from camd.agent.agents import QBCStabilityAgent
from camd.analysis import AnalyzeStability_mod
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
agent = QBCStabilityAgent
agent_params = {
    'ml_algorithm': MLPRegressor,
    'ml_algorithm_params': {'hidden_layer_sizes': (84, 50)},
    'n_query': n_query,
    'n_members': 10,        # Committee size in QBC
    'hull_distance': 0.05,  # Distance to hull to consider a finding as discovery (eV/atom)
    'frac': 0.5             # Fraction of data to choose to form a committee member
    }
analyzer = AnalyzeStability_mod
analyzer_params = {'hull_distance': 0.05}
experiment = ATFSampler
experiment_params = {'dataframe': df}
candidate_data = df
##########################################################
new_loop = Loop(candidate_data, agent, experiment, analyzer,
                agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params,
                create_seed=n_seed)

new_loop.auto_loop(n_iterations=4, timeout=5, initialize=True)
