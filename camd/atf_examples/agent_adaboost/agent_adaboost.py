# Copyright Toyota Research Institute 2019
from camd.loop import Loop

from sklearn.neural_network import MLPRegressor
from camd.agent.agents import AgentStabilityAdaBoost
from camd.analysis import AnalyzeStability
from camd.experiment.base import ATFSampler
from camd.utils.data import load_default_atf_data

##########################################################
# Load dataset and filter by N_species of 2 or less
##########################################################
df = load_default_atf_data()

##########################################################
# Binary stable material discovery 50:50 explore/exploit agent
##########################################################
n_seed = 5000  # Starting sample size - a seed of this size will be randomly chosen.
n_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
agent = AgentStabilityAdaBoost(
    model=MLPRegressor(hidden_layer_sizes=(84, 50)),
    n_query=n_query,
    hull_distance=0.05,
    uncertainty=True,
    exploit_fraction=0.75,
    n_estimators=20
)
analyzer = AnalyzeStability(hull_distance=0.05)
experiment = ATFSampler(dataframe=df)
candidate_data = df

##########################################################
new_loop = Loop(
    candidate_data, agent, experiment, analyzer,
    create_seed=n_seed
)

new_loop.auto_loop(n_iterations=4, timeout=5, initialize=True)
