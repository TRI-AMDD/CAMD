# Copyright Toyota Research Institute 2019
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from camd.utils import aft_loop
from camd.agents import AgentStabilityQBC
from camd.analysis import AnalyzeStability

##########################################################
# Binary stable material discovery QBC based agent recipe
##########################################################
df = pd.read_csv('../oqmd_voro_March25_v2.csv')
df_sub = df[df['N_species'] == 2].sample(frac=0.2) # Downsampling candidates to 20% just for testing!
N_seed = 5000  # Starting sample size
N_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = AgentStabilityQBC
agent_params = {
    'ML_algorithm': MLPRegressor,
    'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
    'N_members': 10,  # Committee size
    'hull_distance': 0.05  # Distance to hull to consider a finding as discovery (eV/atom)
    }
analyzer = AnalyzeStability
analyzer_params = {'hull_distance': 0.05}
##########################################################

path = os.path.abspath('.')
for _ in range(6):
    aft_loop(path, df, df_sub, N_seed, N_query, agent, agent_params, analyzer, analyzer_params)