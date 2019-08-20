# Copyright Toyota Research Institute 2019

import numpy as np
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from copy import deepcopy
from camd.analysis import PhaseSpaceAL, ELEMENTS
from camd.agent.base import HypothesisAgent, QBC
from camd.log import camd_traced

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# TODO: Adaptive N_query and subsampling of candidate space


@camd_traced
class QBCStabilityAgent(HypothesisAgent):

    def __init__(self, candidate_data=None, seed_data=None, N_query=None,
                 pd=None, hull_distance=None, N_species=None, ML_algorithm=None, ML_algorithm_params=None,
                 N_members=None, frac=None, alpha=None, multiprocessing=True):

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.ML_algorithm = ML_algorithm
        self.ML_algorithm_params = ML_algorithm_params
        self.N_members = N_members if N_members else 10
        self.frac = frac if frac else 0.5
        self.alpha = alpha if alpha else 0.5
        self.multiprocessing = multiprocessing
        self.N_species = N_species
        self.cv_score = np.nan

        self.qbc = QBC(N_members=self.N_members, frac=self.frac,
                       ML_algorithm=self.ML_algorithm, ML_algorithm_params=self.ML_algorithm_params)

        super(QBCStabilityAgent, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None, retrain_committee=True):
        if self.N_species:
            self.candidate_data = candidate_data[ candidate_data['N_species'] == self.N_species ]
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data
        if retrain_committee:
            self.qbc.trained = False

        if not self.qbc.trained:
            self.qbc.fit(self.seed_data, self.seed_data['delta_e'], ignore_columns=['Composition',
                                                                                    'N_species', 'delta_e'])
        self.cv_score = self.qbc.cv_score

        # Dropping columns not relevant for ML predictions, but also
        # 'delta_e' column, if exists. The latter is to ensure delta_e does not end up in features if using
        # after the fact data.
        columns_to_drop = ['Composition', 'N_species']
        if 'delta_e' in self.candidate_data:
            columns_to_drop.append('delta_e')

        # QBC makes predictions for Hf and uncertainty on candidate data
        preds, stds = self.qbc.predict(self.candidate_data, ignore_columns=columns_to_drop)
        expected = preds - stds*self.alpha

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


@camd_traced
class AgentStabilityML5(HypothesisAgent):
    """
    An agent that does a certain fraction of full exploration an exploitation in each iteration.
    It will exploit a fraction of N_query options (frac), and explore the rest of its budget.
    """
    def __init__(self, candidate_data=None, seed_data=None, N_query=None,
                 pd=None, hull_distance=None, N_species=None, ML_algorithm=None, ML_algorithm_params=None,
                 frac=None, multiprocessing=True):

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.ML_algorithm = ML_algorithm
        self.ML_algorithm_params = ML_algorithm_params
        self.multiprocessing = multiprocessing
        self.frac = frac if frac else 0.5
        self.cv_score = np.nan
        self.N_species = N_species

        super(AgentStabilityML5, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        if self.N_species:
            self.candidate_data = candidate_data[ candidate_data['N_species'] == self.N_species ]
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data
        overall_model = self.ML_algorithm(**self.ML_algorithm_params)
        X = self.seed_data.drop(['Composition', 'N_species', 'delta_e'], axis=1)

        from sklearn.preprocessing import StandardScaler
        overall_scaler = StandardScaler()
        X = overall_scaler.fit_transform(X, self.seed_data['delta_e'])
        overall_model.fit(X, self.seed_data['delta_e'])

        from sklearn.model_selection import cross_val_score, KFold
        cv_score = cross_val_score(overall_model, X, self.seed_data['delta_e'],
                                   cv=KFold(5, shuffle=True), scoring='neg_mean_absolute_error')
        self.cv_score = np.mean(cv_score)*-1

        # Dropping columns not relevant for ML predictions, but also
        # 'delta_e' column, if exists. The latter is to ensure delta_e does not end up in features if using
        # after the fact data.
        columns_to_drop = ['Composition', 'N_species']
        if 'delta_e' in self.candidate_data:
            columns_to_drop.append('delta_e')
        cand_X = self.candidate_data.drop(columns_to_drop, axis=1)
        cand_X = overall_scaler.transform(cand_X)
        expected = overall_model.predict(cand_X)

        # this is just curbing outrageously negative predictions
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

        sorted_stabilities = sorted(ml_stable, key=lambda x: x.stability)

        # Exploitation part:
        to_compute = sorted_stabilities[:int(self.N_query * self.frac)]
        remaining = sorted_stabilities[int(self.N_query * self.frac):]

        # Exploration part:
        np.random.shuffle(remaining)
        to_compute += remaining[:int(self.N_query * (1.0-self.frac))]

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


class GaussianProcessStabilityAgent(HypothesisAgent):
    """
    Simple Gaussian Process Regressor based Stability Agent
    """
    def __init__(self, candidate_data=None, seed_data=None, N_query=None,
                 pd=None, hull_distance=None, N_species=None, alpha=None, multiprocessing=True):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.alpha = alpha if alpha else 0.5
        self.multiprocessing = multiprocessing
        self.N_species = N_species
        self.cv_score = np.nan
        self.GP = GaussianProcessRegressor(kernel=C(1) * RBF(1), alpha=0.002)

        super(GaussianProcessStabilityAgent, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        if self.N_species:
            self.candidate_data = candidate_data[candidate_data['N_species'] == self.N_species]
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data

        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        X = self.seed_data.drop(columns_to_drop, axis=1)
        y = self.seed_data['delta_e']

        from sklearn.preprocessing import StandardScaler

        steps = [('scaler', StandardScaler()), ('GP', self.GP)]
        cv_pipeline = Pipeline(steps)
        self.cv_score = np.mean(-1.0 * cross_val_score(cv_pipeline, X, y,
                                                       cv=KFold(3, shuffle=True), scoring='neg_mean_absolute_error'))

        steps = [('scaler', StandardScaler()), ('GP', self.GP)]
        overall_pipeline = Pipeline(steps)
        overall_pipeline.fit(X, y)

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        preds, stds = overall_pipeline.predict(self.candidate_data.drop(columns_to_drop, axis=1),
                                               **{'return_std': True})
        expected = preds - stds * self.alpha

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


try:
    import gpflow
    gpflow_loaded = True
except ImportError:
    gpflow_loaded = False

class SVGProcessStabilityAgent(HypothesisAgent):
    """
    Stochastic variational gaussian process stability agent for Big Data.

    The computational complexity of this algorithm scales as O(M^3) compared to O(N^3) of standard GP,
    where N is the number of data points and M is the number of inducing points (M<<N).

    The default parameters are optimized to deliver a compromise between compute-time and model
    accuracy for data sets with up to 25 to 40 thousand examples (e.g. the ICSD seed data). For bigger systems,
    parameter M may need to be reduced. For smaller systems, it can be increased, if higher accuracy is desired.
    Inducing point locations are determined using k-means clustering.

    References:
    Hensman, James, Nicolo Fusi, and Neil D. Lawrence. "Gaussian processes for big data."
                                        Uncertainty in Artificial Intelligence (2013).
    Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
                                        arXiv preprint arXiv:1412.6980 (2014).

    """
    def __init__(self, candidate_data=None, seed_data=None, N_query=None,
                 pd=None, hull_distance=None, N_species=None, alpha=None, M=None, multiprocessing=True):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.hull_distance = hull_distance if hull_distance else 0.0
        self.N_query = N_query if N_query else 1
        self.pd = pd
        self.alpha = alpha if alpha else 0.5
        self.M = M if M else 700
        self.multiprocessing = multiprocessing
        self.N_species = N_species
        self.cv_score = np.nan

        self.kernel = gpflow.kernels.RBF(273)*gpflow.kernels.Constant(273)
        self.mean_f = gpflow.mean_functions.Constant()
        self.GP = GaussianProcessRegressor(kernel=C(1) * RBF(1), alpha=0.002)

        super(SVGProcessStabilityAgent, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        if self.N_species:
            self.candidate_data = candidate_data[candidate_data['N_species'] == self.N_species]
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data

        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        X = self.seed_data.drop(columns_to_drop, axis=1)
        y = self.seed_data['delta_e']

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.model_selection import train_test_split


        ## Let's test model performance first.
        # Note we avoid doing CV to reduce compute time. We simply do a 1-way split 80:20 (train:test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # We do a k-means clustering and use cluster centers as inducing points.
        cls = MiniBatchKMeans(n_clusters=self.M, batch_size=200)
        cls.fit(X_train_scaled)
        Z = cls.cluster_centers_
        _y = np.array(y_train.to_list())
        mu = np.mean(_y)
        sig = np.std(_y)
        model = gpflow.models.SVGP(X_train_scaled, ((_y - mu)/sig).reshape(-1, 1), self.kernel,
                                gpflow.likelihoods.Gaussian(), Z, mean_function=self.mean_f, minibatch_size=100)
        logger = self.run_adam(model, gpflow.test_util.notebook_niter(20000))

        pred_y, pred_v = model.predict_y(scaler.transform(X_test))
        pred_y = pred_y * sig + mu
        self.pred_y = pred_y
        self.pred_std = pred_v**0.5
        self.cv_score = np.mean(np.abs(pred_y - y_test.to_numpy().reshape(-1, 1)))

        #overall model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cls = MiniBatchKMeans(n_clusters=self.M, batch_size=200)
        cls.fit(X_train_scaled)
        Z = cls.cluster_centers_
        _y = np.array(y.to_list())
        mu = np.mean(_y)
        sig = np.std(_y)
        model = gpflow.models.SVGP(X_scaled, ((_y - mu)/sig).reshape(-1, 1), self.kernel,
                                gpflow.likelihoods.Gaussian(), Z, mean_function=self.mean_f, minibatch_size=100)
        logger = self.run_adam(model, gpflow.test_util.notebook_niter(20000))


        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        pred_y, pred_y =  model.predict_y(scaler.transform( self.candidate_data.drop(columns_to_drop, axis=1)))
        expected = pred_y - pred_y**0.5 * self.alpha

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

    def run_adam(self, model, iterations):
        """
        Adam optimizer as implemented in:
        https://github.com/GPflow/GPflow/blob/develop/doc/source/notebooks/advanced/gps_for_big_data.ipynb
        """
        # Create an Adam Optimiser action
        adam = gpflow.train.AdamOptimizer().make_optimize_action(model)
        # Create a Logger action
        self.logger = self.Logger(model)
        actions = [adam, self.logger]
        # Create optimisation loop that interleaves Adam with Logger
        loop = gpflow.actions.Loop(actions, stop=iterations)()
        # Bind current TF session to model
        model.anchor(model.enquire_session())
        return self.logger

    class Logger(gpflow.actions.Action):
        """
        Logger class as implemented in
        https://github.com/GPflow/GPflow/blob/develop/doc/source/notebooks/advanced/gps_for_big_data.ipynb
        """
        def __init__(self, model):
            self.model = model
            self.logf = []

        def run(self, ctx):
            if (ctx.iteration % 10) == 0:
                # Extract likelihood tensor from Tensorflow session
                likelihood = - ctx.session.run(self.model.likelihood_tensor)
                # Append likelihood value to list
                self.logf.append(likelihood)