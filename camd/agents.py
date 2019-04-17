# Copyright Toyota Research Institute 2019

import numpy as np
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from copy import deepcopy
from camd.analysis import PhaseSpaceAL, ELEMENTS
from camd.hypothesis import HypothesisBase, QBC

# TODO: Adaptive N_query and subsampling of candidate space


class AgentStabilityQBC(HypothesisBase):

    def __init__(self, candidate_data, seed_data, N_query=None,
                 pd=None, hull_distance=None, ML_algorithm=None, ML_algorithm_params=None,
                 N_members=None, frac=None, multiprocessing=True):

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.ML_algorithm = ML_algorithm
        self.ML_algorithm_params = ML_algorithm_params
        self.N_members = N_members if N_members else 10
        self.frac = frac if frac else 0.5
        self.multiprocessing = multiprocessing

        self.cv_score = np.nan

        self.qbc = QBC(N_members=self.N_members, frac=self.frac,
                       ML_algorithm=self.ML_algorithm, ML_algorithm_params=self.ML_algorithm_params)

        super(AgentStabilityQBC, self).__init__()

    def hypotheses(self, retrain_committee=False):
        if retrain_committee:
            self.qbc.trained = False

        if not self.qbc.trained:
            self.qbc.fit(self.seed_data, self.seed_data['delta_e'], ignore_columns=['Composition', 'N_species', 'delta_e'])
        self.cv_score = self.qbc.cv_score

        # QBC makes predictions for Hf and uncertainty on candidate data
        preds, stds = self.qbc.predict(self.candidate_data)
        expected = preds - stds

        # This is just curbing outrageously negative predictions
        for i in range(len(expected)):
            if expected[i] < -6.0:
                expected[i] = -6.0

        # Get estimated stabilities from ML predictions
        # For that, let's create Phases from candidates
        candidate_phases = []
        _c = 0
        for data in self.candidate_data.iterrows():
            candidate_phases.append(
                Phase(data[1]['Composition'], energy=expected[_c], per_atom=True, description=data[0]))
            _c += 1

        # We take the existing phase data for seed phases, add candidate phases, and compute stabilities
        if not self.pd:
            self.get_pd()

        pd_ml = deepcopy(self.pd)
        pd_ml.add_phases(candidate_phases)
        space_ml = PhaseSpaceAL(bounds=ELEMENTS, data=pd_ml)
        if self.multiprocessing:
            space_ml.compute_stabilities_multi(candidate_phases)
        else:
            space_ml.compute_stabilities_mod(candidate_phases)

        ml_stabilities = []
        for _p in candidate_phases:
            ml_stabilities.append(_p.stability)

        # Now let's find the most stable ones upto N_query within hull_distance
        ml_stabilities = np.array(ml_stabilities, dtype=float)
        ml_stable = np.array(candidate_phases)[ml_stabilities <= self.hull_distance]
        to_compute = sorted(ml_stable, key=lambda x: x.stability)[:self.N_query]

        self.indices_to_compute = [i.description for i in to_compute]

        return self.indices_to_compute

    def get_pd(self):
        self.pd = PhaseData()
        phases = []
        for data in self.seed_data.iterrows():
            phases.append(Phase(data[1]['Composition'], energy=data[1]['delta_e'], per_atom=True, description=data[0]))
        for el in ELEMENTS:
            phases.append(Phase(el, 0.0, per_atom=True))
        self.pd.add_phases(phases)


class AgentRandom(HypothesisBase):
    """
    Baseline agent: Randomly picks next experiments
    """
    def __init__(self, candidate_data, seed_data, N_query=None, pd=None, hull_distance=None):

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.cv_score = np.nan
        super(AgentRandom, self).__init__()

    def hypotheses(self):
        indices_to_compute = []
        for data in self.candidate_data.iterrows():
            indices_to_compute.append(data[0])
        a = np.array(indices_to_compute)
        np.random.shuffle(a)
        indices_to_compute = a[:self.N_query].tolist()
        return indices_to_compute