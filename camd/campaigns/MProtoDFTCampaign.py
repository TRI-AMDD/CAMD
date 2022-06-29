"""
Script test of MProtoDFTCampagin
"""
from monty.tempfile import ScratchDir
from fireworks import LaunchPad
from sklearn.neural_network import MLPRegressor
from camd import __version__
from camd.domain import StructureDomain, heuristic_setup
from camd.agent.m3gnet import M3GNetStabilityAgent
from camd.agent.stability import AgentStabilityAdaBoost
from camd.agent.base import SequentialAgent
from camd.campaigns.structure_discovery import ProtoDFTCampaign
from camd.experiment.dft import AtomateExperiment
from camd.analysis import StabilityAnalyzer

from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet


def load_candidate(chemsys, n_max_atoms=20):
    """
    Load candidate data
    Args:
        chemsys (str): chemical system
        n_max_atoms (int): number of maximum atoms
    """
    element_list = chemsys.split("-")
    max_coeff, charge_balanced = heuristic_setup(element_list)
    domain = StructureDomain.from_bounds(
        element_list,
        charge_balanced=charge_balanced,
        n_max_atoms=n_max_atoms,
        **{"grid": range(1, max_coeff)}
    )
    candidate_data = domain.candidates()
    return candidate_data


def load_seed(chemsys="Si"):
    """
    Helper function
    Returns:
        list of the MP entries (w/ structure)
        in the chemsys
    """
    with MPRester() as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys, inc_structure=True)
    return entries


if __name__ == "__main__":
    DB_FILE = ""
    LPAD_FILE = ""
    CHEMSYS = "Si"
    seed_data = load_seed(CHEMSYS)
    exp_data = load_candidate(CHEMSYS)
    model = M3GNet(is_intensive=False)
    m3gnetagent = M3GNetStabilityAgent(m3gnet=model, hull_distance=2.0)
    adagent = AgentStabilityAdaBoost(
        model=MLPRegressor(hidden_layer_sizes=(20,)),
        n_query=3,
        hull_distance=100.0,
        exploit_fraction=1.0,
        uncertainty=True,
        alpha=0.5,
        diversify=True,
        n_estimators=4,
    )
    agent = SequentialAgent(agents=[adagent, m3gnetagent])
    analyzer = StabilityAnalyzer(hull_distance=0.2)
    lpad = LaunchPad.from_file(LPAD_FILE)
    lpad.auto_load()
    experiment = AtomateExperiment(lpad, DB_FILE, poll_time=30, launch_from_local=False)
    with ScratchDir("."):
        campaign = ProtoDFTCampaign(
            candidate_data=exp_data.drop("delta_e", axis=1),
            agent=agent,
            experiment=experiment,
            analyzer=analyzer,
            seed_data=seed_data,
            heuristic_stopper=5,
        )
        campaign.autorun()
