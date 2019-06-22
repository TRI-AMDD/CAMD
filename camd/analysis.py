# Copyright Toyota Research Institute 2019

import abc

import numpy as np
from camd import tqdm
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from qmpy.analysis.thermodynamics.space import PhaseSpace
import multiprocessing
from pymatgen import Composition

ELEMENTS = ['Ru', 'Re', 'Rb', 'Rh', 'Be', 'Ba', 'Bi', 'Br', 'H', 'P', 'Os', 'Ge', 'Gd', 'Ga', 'Pr', 'Pt', 'Pu', 'C',
            'Pb', 'Pa', 'Pd', 'Xe', 'Pm', 'Ho', 'Hf', 'Hg', 'He', 'Mg', 'K', 'Mn', 'O', 'S', 'W', 'Zn', 'Eu', 'Zr',
            'Er', 'Ni', 'Na', 'Nb', 'Nd', 'Ne', 'Np', 'Fe', 'B', 'F', 'Sr', 'N', 'Kr', 'Si', 'Sn', 'Sm', 'V', 'Sc',
            'Sb', 'Se', 'Co', 'Cl', 'Ca', 'Ce', 'Cd', 'Tm', 'Cs', 'Cr', 'Cu', 'La', 'Li', 'Tl', 'Lu', 'Th', 'Ti', 'Te',
            'Tb', 'Tc', 'Ta', 'Yb', 'Dy', 'I', 'U', 'Y', 'Ac', 'Ag', 'Ir', 'Al', 'As', 'Ar', 'Au', 'In', 'Mo']


#TODO: Eval Performance = start / stop?

class AnalyzerBase(abc.ABC):
    @abc.abstractmethod
    def analyze(self):
        """
        Performs the analysis procedure associated with the analyzer

        # TODO: I'm not yet sure what we might want to do here
        #       in terms of enforcing a result contract
        Returns:
            Some arbitrary result

        """

    @abc.abstractmethod
    def present(self):
        """
        Formats the analysis into a some presentation-oriented
        document

        Returns:
            json document for presentation, e. g. on a web frontend

        """


class AnalyzeStability(AnalyzerBase):
    """
    This the original stability analyzer. It will be replaced with AnalyzeStability_mod in the near future as
    we finish adding the new functionality to the new class and fully test.
    """
    def __init__(self, df=None, new_result_ids=None, hull_distance=None, multiprocessing=True):
        self.df = df
        self.new_result_ids = new_result_ids
        self.hull_distance = hull_distance if hull_distance else 0.05
        self.multiprocessing = multiprocessing
        super(AnalyzeStability, self).__init__()

    def analyze(self, df=None, new_result_ids=None, all_result_ids=None):
        self.df = df
        self.new_result_ids = new_result_ids
        phases = []
        for data in self.df.iterrows():
            phases.append(Phase(data[1]['Composition'], energy=data[1]['delta_e'], per_atom=True, description=data[0]))
        for el in ELEMENTS:
            phases.append(Phase(el, 0.0, per_atom=True))

        pd = PhaseData()
        pd.add_phases(phases)
        space = PhaseSpaceAL(bounds=ELEMENTS, data=pd)

        if self.multiprocessing:
            space.compute_stabilities_multi()
        else:
            space.compute_stabilities_mod()

        # Add dtype so that None values can be compared
        stabilities_of_space_uids = np.array([p.stability for p in space.phases],
                                             dtype=np.float) <= self.hull_distance
        stabilities_of_new = {}
        for _p in space.phases:
            if _p.description in self.new_result_ids:
                stabilities_of_new[_p.description] = _p.stability

        self.stabilities_of_new = stabilities_of_new
        stabilities_of_new_uids = np.array([stabilities_of_new[uid] for uid in self.new_result_ids],
                                           dtype=np.float) <= self.hull_distance

        # array of bools for stable vs not for new uids, and all experiments, respectively
        return stabilities_of_new_uids, stabilities_of_space_uids

    def present(self):
        pass


class AnalyzeStability_mod(AnalyzerBase):
    def __init__(self, df=None, new_result_ids=None, hull_distance=None, multiprocessing=True, entire_space=False):
        self.df = df
        self.new_result_ids = new_result_ids
        self.hull_distance = hull_distance if hull_distance else 0.05
        self.multiprocessing = multiprocessing
        self.entire_space = entire_space
        super(AnalyzeStability_mod, self).__init__()

    def analyze(self, df=None, new_result_ids=None, all_result_ids=None):
        self.df = df.drop_duplicates(keep='last').dropna()
        # Note some of id's in all_result_ids may not have corresponding experiment, if those exps. failed.
        self.all_result_ids = all_result_ids
        self.new_result_ids = new_result_ids

        if not self.entire_space:
            # This option constraints the phase space to that of the target compounds. This should be more efficient
            # when searching in a specified chemistry, less efficient if larger spaces are being scanned without chemistry
            # focus.

            # Note this line needs to be fixed later to be compatible with later versions of pandas (i.e.
            # b/c all_result_ids may contain things not in df currently (b/c of failed experiments).
            # We should test comps = self.df.loc[self.df.index.intersection(all_result_ids)]

            comps = self.df.loc[all_result_ids]['Composition'].dropna()
            system_elements = []
            for comp in comps:
                system_elements += list(Composition(comp).as_dict().keys())
            elems = set(system_elements)
            ind_to_include = []
            for ind in self.df.index:
                if set(Composition(self.df.loc[ind]['Composition']).as_dict().keys()).issubset(elems):
                    ind_to_include.append(ind)
            _df = self.df.loc[ind_to_include]
        else:
            _df = self.df

        phases = []
        for data in _df.iterrows():
            phases.append(Phase(data[1]['Composition'], energy=data[1]['delta_e'], per_atom=True, description=data[0]))
        for el in ELEMENTS:
            phases.append(Phase(el, 0.0, per_atom=True))

        pd = PhaseData()
        pd.add_phases(phases)
        space = PhaseSpaceAL(bounds=ELEMENTS, data=pd)

        if all_result_ids:
            all_new_phases = [p for p in space.phases if p.description in all_result_ids]
        else:
            all_new_phases = None

        if self.multiprocessing:
            space.compute_stabilities_multi(phases_to_evaluate=all_new_phases)
        else:
            space.compute_stabilities_mod(phases_to_evaluate=all_new_phases)

        # Add dtype so that None values can be compared
        stabilities_of_space_uids = np.array([p.stability for p in all_new_phases],
                                             dtype=np.float) <= self.hull_distance
        stabilities_of_new = {}
        for _p in all_new_phases:
            if _p.description in self.new_result_ids:
                stabilities_of_new[_p.description] = _p.stability

        self.stabilities_of_new = stabilities_of_new

        stabilities_of_new_uids = []
        for uid in self.new_result_ids:
            if uid in stabilities_of_new:
                stabilities_of_new_uids.append(stabilities_of_new[uid])
            else:
                stabilities_of_new_uids.append(np.nan)
        stabilities_of_new_uids = np.array(stabilities_of_new_uids, dtype=np.float) <= self.hull_distance

        # stabilities_of_new_uids = np.array([stabilities_of_new[uid] for uid in self.new_result_ids],
        #                                    dtype=np.float) <= self.hull_distance

        # array of bools for stable vs not for new uids, and all experiments, respectively
        return stabilities_of_new_uids, stabilities_of_space_uids

    def present(self):
        pass

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

        for p in tqdm(list(self.phase_dict.values())):
            if p.stability is None:  # for low e phases, we only need to eval stability if it doesn't exist
                try:
                    p.stability = p.energy - self.gclp(p.unit_comp)[0]
                except:
                    print(p)
                    p.stability = np.nan

        # will only do requested phases for things not in phase_dict
        for p in tqdm(phases_to_evaluate):
            if p not in list(self.phase_dict.values()):
                if p.name in self.phase_dict:
                    p.stability = p.energy - self.phase_dict[p.name].energy + self.phase_dict[p.name].stability
                else:
                    try:
                        p.stability = p.energy - self.gclp(p.unit_comp)[0]
                    except:
                        print(p)
                        p.stability = np.nan

    def compute_stabilities_multi(self, phases_to_evaluate=None, ncpus=multiprocessing.cpu_count()):
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

        # Creating a map from entry uid to index of entry in the current list of phases in space.
        self.uid_to_phase_ind = dict([(self.phases[i].description, i) for i in range(len(self.phases))])

        phase_dict_list = list(self.phase_dict.values())
        _result_list1 = parmap(self._multiproc_help1,  phase_dict_list, nprocs=ncpus)
        for i in range(len(phase_dict_list)):
            self.phase_dict[phase_dict_list[i].name].stability = _result_list1[i]

        _result_list2 = parmap(self._multiproc_help2, phases_to_evaluate, nprocs=ncpus)
        for i in range(len(phases_to_evaluate)):
            # we will use the uid_to_phase_ind create above to be able to map results of parmap to self.phases
            ind = self.uid_to_phase_ind[phases_to_evaluate[i].description]
            self.phases[ind].stability = _result_list2[i]


    def _multiproc_help1(self, p):
        if p.stability is None:  # for low e phases, we only need to eval stability if it doesn't exist
            try:
                p.stability = p.energy - self.gclp(p.unit_comp)[0]
            except:
                print(p)
                p.stability = np.nan
        return p.stability

    def _multiproc_help2(self, p):
        if p not in list(self.phase_dict.values()):
            if p.name in self.phase_dict:
                p.stability = p.energy - self.phase_dict[p.name].energy + self.phase_dict[p.name].stability
            else:
                try:
                    p.stability = p.energy - self.gclp(p.unit_comp)[0]
                except:
                    print(p)
                    p.stability = np.nan
        return p.stability


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
