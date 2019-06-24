# Copyright Toyota Research Institute 2019
from camd.loop import Loop
import pandas as pd

from sklearn.neural_network import MLPRegressor
from camd.agent.agents import AgentStabilityML5
from camd.analysis import AnalyzeStability_mod
from camd.experiment.base import ATFSampler
from camd.utils.s3 import cache_s3_objs
from camd import S3_CACHE
import os

cache_s3_objs(['camd/shared-data/oqmd_1.2_voronoi_magpie_fingerprints.pickle'])
##########################################################
# Binary stable material discovery 50:50 explore/exploit agent
##########################################################
df = pd.read_pickle(os.path.join(S3_CACHE,
                              'camd/shared-data/oqmd_1.2_voronoi_magpie_fingerprints.pickle')).sample(frac=0.2)
N_seed = 5000  # Starting sample size
N_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = AgentStabilityML5
agent_params = {
    'ML_algorithm': MLPRegressor,
    'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
    'N_query': N_query, 'N_species': 2,
    'hull_distance': 0.05,  # Distance to hull to consider a finding as discovery (eV/atom)
    'frac': 0.5 # Fraction to exploit (rest will be explored -- randomly picked)
    }
analyzer = AnalyzeStability_mod
analyzer_params = {'hull_distance': 0.05}
experiment = ATFSampler
experiment_params = {'params': {'dataframe': df}}
candidate_data = df
##########################################################
new_loop = Loop(candidate_data, agent, experiment, analyzer,
               agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params,
               create_seed=N_seed)

new_loop.auto_loop(n_iterations=4, timeout=5, initialize=True)