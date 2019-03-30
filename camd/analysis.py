# Copyright Toyota Research Institute 2019

import numpy as np
from tqdm import tqdm
from qmpy import PhaseSpace, Phase, PhaseData
from abc import ABCMeta

ELEMENTS = ["Pr", "Ru", "Th", "Pt", "Ni", "S", "Na", "Nb", "Nd", "C", "Li", "Pb", "Y", "Tl", "Lu", "Rb", "Ti", "Np",
            "Te", "Rh", "Tc", "La", "Ta", "Be", "Sr", "Sm", "Ba", "Tb", "Yb", "Bi", "Re", "Pu", "Fe", "Br", "Dy", "Pd",
            "Hf", "Hg", "Ho", "Mg", "B", "Pm", "P", "F", "I", "H", "K", "Mn", "Ac", "O", "N", "Eu", "Si", "U", "Sn",
            "W", "V", "Sc", "Sb", "Mo", "Os", "Se", "Zn", "Co", "Ge", "Ag", "Cl", "Ca", "Ir", "Al", "Ce", "Cd", "Pa",
            "As", "Gd", "Au", "Cu", "Ga", "In", "Cs", "Cr", "Tm", "Zr", "Er"]


#TODO: Eval Performance = start / stop?

class AnalysisBase:
    __metaclass__ = ABCMeta

    def hypotheses(self):
        pass


class AnalyzeStability(AnalysisBase):
    def __init__(self, df, new_result_ids, hull_distance=None):
        self.df = df
        self.new_result_ids = new_result_ids
        self.hull_distance = hull_distance if hull_distance else 0.05
        super(AnalyzeStability, self).__init__()

    def analysis(self):
        phases = []
        for data in self.df.iterrows():
            phases.append(Phase(data[1]['Composition'], energy=data[1]['delta_e'], per_atom=True, description=data[0]))
        for el in ELEMENTS:
            phases.append(Phase(el, 0.0, per_atom=True))

        pd = PhaseData()
        pd.add_phases(phases)
        space = PhaseSpaceAL(bounds=ELEMENTS, data=pd)
        space.compute_stabilities_mod()

        stabilities_of_space_uids = np.array([p.stability for p in space.phases]) <= self.hull_distance

        stabilities_of_new = {}
        for _p in space.phases:
            if _p.description in self.new_result_ids:
                stabilities_of_new[_p.description] = _p.stability

        stabilities_of_new_uids = np.array([stabilities_of_new[uid] for uid in self.new_result_ids]) <= self.hull_distance

        # array of bools for stable vs not for new uids, and all experiments, respectively
        return stabilities_of_new_uids, stabilities_of_space_uids


class PhaseSpaceAL(PhaseSpace):
    """
    Modified qmpy.PhaseSpace for GCLP based stabiltiy computations
    TODO: basic multithread or Gurobi for gclp
    """

    def compute_stabilities_mod(self, phases_to_evaluate=None):
        """
        Calculate the stability for every Phase.
        Keyword Arguments:
            phases:
                List of Phases. If None, uses every Phase in PhaseSpace.phases
            save:
                If True, save the value for stability to the database.
            new_only:
                If True, only compute the stability for Phases which did not
                import a stability from the OQMD. False by default.
        """

        if phases_to_evaluate is None:
            phases_to_evaluate = self.phases

        for p in tqdm(self.phase_dict.values()):
            if p.stability is None:  # for low e phases, we only need to eval stability if it doesn't exist
                try:
                    p.stability = p.energy - self.gclp(p.unit_comp)[0]
                except:
                    print p
                    p.stability = np.nan

        # will only do requested phases for things not in phase_dict
        for p in tqdm(phases_to_evaluate):
            if p not in self.phase_dict.values():
                if p.name in self.phase_dict:
                    p.stability = p.energy - self.phase_dict[p.name].energy + self.phase_dict[p.name].stability
                else:
                    try:
                        p.stability = p.energy - self.gclp(p.unit_comp)[0]
                    except:
                        print p
                        p.stability = np.nan