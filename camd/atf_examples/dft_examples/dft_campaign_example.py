from camd.domain import StructureDomain
from camd.loop import Loop
from camd.agent.stability import QBCStabilityAgent
from camd.analysis import AnalyzeStability_mod
from camd.experiment.dft import OqmdDFTonMC1
from sklearn.neural_network import MLPRegressor

# Let's create our search domain as Ir-Fe-O ternary. We restrict our search to structures with max 10 atoms.
# We further restrict the possible stoichiometry coefficients to integers to [1,4).
domain = StructureDomain.from_bounds(["Ir", "Fe", "O"], charge_balanced=True, n_max_atoms = 10, **{"grid": range(1,4)})
candidate_data = domain.candidates()
structure_dict = domain.hypo_structures_dict

# Setup the loop for this campaign.

agent = QBCStabilityAgent           # We use a query-by-committee (QBC) based agent
agent_params = {                    # Parameters of the agent
    'ML_algorithm': MLPRegressor,   # We use simple fully connected neural network as our regressor
    'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
    'N_query': 3,                   # Agent is allowed 3 experiments per iteration
    'N_members': 10,                # Committee size for QBC
    'hull_distance': 0.1,   # Distance to hull to consider a finding as discovery (eV/atom)
    'frac': 0.5
    }
analyzer = AnalyzeStability_mod     # Analyzer for stability
analyzer_params = {'hull_distance': 0.1}
experiment = OqmdDFTonMC1           # This is the Experiment method to run OQMD compatible DFT on AWS-MC1
experiment_params = {'structure_dict': structure_dict,  # Parameters of this experiment class include structures.
                     'candidate_data': candidate_data, 'timeout': 30000}

# Loop class puts all the above pieces together.
new_loop = Loop(candidate_data, agent, experiment, analyzer,
                agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params)

# Let's start the campaign!
new_loop.auto_loop_in_directories(n_iterations=3, timeout=1, monitor=True, initialize=True, with_icsd=True)