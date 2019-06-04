# Copyright Toyota Research Institute 2019
from camd.loop import Loop
import pandas as pd

from camd.agent.base import RandomAgent
from camd.analysis import AnalyzeStability
from camd.experiment.base import ATFSampler
from camd.utils.s3 import cache_s3_objs
from camd import S3_CACHE
import os

cache_s3_objs(['camd/shared-data/oqmd_voro_March25_v2.csv'])

##########################################################
# Binary stable material discovery QBC based agent recipe
##########################################################
df = pd.read_csv(os.path.join(S3_CACHE, 'camd/shared-data/oqmd_voro_March25_v2.csv'))
N_seed = 5000  # Starting sample size
N_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = RandomAgent
agent_params = {'hull_distance': 0.05, 'N_query': N_query, 'N_species': 2}  # Distance to hull to consider a finding as discovery (eV/atom)
analyzer = AnalyzeStability
analyzer_params = {'hull_distance': 0.05}
experiment = ATFSampler
experiment_params = {'dataframe': df}
candidate_data = df
path = '.'
##########################################################
new_loop = Loop(path, candidate_data, agent, experiment, analyzer,
               agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params,
               create_seed=N_seed)

new_loop.auto_loop(n_iterations=4, timeout=5)