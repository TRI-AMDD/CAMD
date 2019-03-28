# Copyright Toyota Research Institute 2019

import numpy as np
from tqdm import tqdm
from qmpy import PhaseSpace, Phase, PhaseData

ELEMENTS = ["Pr", "Ru", "Th", "Pt", "Ni", "S", "Na", "Nb", "Nd", "C", "Li", "Pb", "Y", "Tl", "Lu", "Rb", "Ti", "Np",
            "Te", "Rh", "Tc", "La", "Ta", "Be", "Sr", "Sm", "Ba", "Tb", "Yb", "Bi", "Re", "Pu", "Fe", "Br", "Dy", "Pd",
            "Hf", "Hg", "Ho", "Mg", "B", "Pm", "P", "F", "I", "H", "K", "Mn", "Ac", "O", "N", "Eu", "Si", "U", "Sn",
            "W", "V", "Sc", "Sb", "Mo", "Os", "Se", "Zn", "Co", "Ge", "Ag", "Cl", "Ca", "Ir", "Al", "Ce", "Cd", "Pa",
            "As", "Gd", "Au", "Cu", "Ga", "In", "Cs", "Cr", "Tm", "Zr", "Er"]


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


def get_stabilities_from_data(df, uids, hull_distance=0.0):

    phases = []
    for data in df.iterrows():
        phases.append(Phase(data[1]['Composition'], energy=data[1]['delta_e'], per_atom=True, description=data[0]))
    for el in ELEMENTS:
        phases.append(Phase(el, 0.0, per_atom=True))

    pd = PhaseData()
    pd.add_phases(phases)
    space = PhaseSpaceAL(bounds=ELEMENTS, data=pd)
    space.compute_stabilities_mod()
    space_stabilities = np.array([p.stability for p in space.phases])
    n_stable = np.sum(space_stabilities <= hull_distance)

    stabilities_of_uids = {}
    for _p in space.phases:
        if _p.description in uids:
            stabilities_of_uids[_p.description] = _p.stability

    return stabilities_of_uids, space_stabilities, n_stable