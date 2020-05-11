from camd.domain import StructureDomain
from camd.campaigns.base import Campaign
from camd.agent.stability import QBCStabilityAgent
from camd.analysis import StabilityAnalyzer
from camd.experiment.dft import OqmdDFTonMC1
from sklearn.neural_network import MLPRegressor

# Let's create our search domain as Ir-Fe-O ternary. We restrict our search to structures with max 10 atoms.
# We further restrict the possible stoichiometry coefficients to integers to [1,4).

domain = StructureDomain.from_bounds(["Ir", "Fe", "O"], charge_balanced=True, n_max_atoms = 10, **{"grid": range(1,4)})
candidate_data = domain.candidates()
structure_dict = domain.hypo_structures_dict

# Setup the loop for this campaign.

agent = QBCStabilityAgent(MLPRegressor(hidden_layer_sizes=(84, 50)),
                          n_query=3,
                          hull_distance=0.1,
                          training_fraction=0.5,
                          n_members=10)
analyzer = StabilityAnalyzer(hull_distance=0.05)
experiment = OqmdDFTonMC1           # This is the Experiment method to run OQMD compatible DFT on AWS-MC1
experiment_params = {'structure_dict': structure_dict,  # Parameters of this experiment class include structures.
                     'candidate_data': candidate_data, 'timeout': 30000}

# Campaign class puts all the above pieces together.
new_loop = Campaign(candidate_data, agent, experiment, analyzer,
                    agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params)

# Let's start the campaign!
new_loop.auto_loop_in_directories(n_iterations=3, timeout=1, monitor=True, initialize=True, with_icsd=True)