# Copyright Toyota Research Institute 2019

import time
import abc
from copy import deepcopy
from multiprocessing import cpu_count

import gpflow
import numpy as np
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from camd.analysis import PhaseSpaceAL, ELEMENTS
from camd.agent.base import HypothesisAgent, QBC
from camd.log import camd_traced

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cluster import MiniBatchKMeans

# TODO: Adaptive N_query and subsampling of candidate space
# TODO: Exploit/Explore general method
# TODO: General uncertainty handling?


class StabilityAgent(HypothesisAgent, metaclass=abc.ABCMeta):
    """
    The StabilityAgent is a mixin abstract class which contains
    initialization parameters and methods common to every agent
    which is responsible for making decisions about stability.
    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool, int): whether to use multiprocessing
                for phase stability analysis, if an int, sets the n_jobs
                parameter as well.  If a bool, sets n_jobs to cpu_count()
                if True and n_jobs to 1 if false.
        """
        super().__init__()
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        self.hull_distance = hull_distance
        self.pd = pd

        # TODO: Probably should just use a single parameter here
        self.multiprocessing = multiprocessing
        if isinstance(self.multiprocessing, bool):
            if self.multiprocessing:
                self.n_jobs = cpu_count()
            else:
                self.n_jobs = 1
        elif isinstance(self.multiprocessing, int) and self.multiprocessing > 0:
            self.n_jobs = self.multiprocessing
        else:
            self.n_jobs = 1

        # These might be able to go into the base class
        self.cv_score = np.nan
        self.indices_to_compute = None

    def get_pd(self):
        """
        Refresh the phase diagram associated with the seed_data

        Returns:
            None
        """
        self.pd = PhaseData()
        phases = [Phase(row['Composition'], energy=row['delta_e'],
                        per_atom=True, description=row_index)
                  for row_index, row in self.seed_data.iterrows()]
        phases.extend([Phase(el, 0.0, per_atom=True) for el in ELEMENTS])
        self.pd.add_phases(phases)
        return self.pd

    def update_candidate_stabilities(self, formation_energies,
                                     sort=True, floor=-6.0):
        """

        Args:
            formation_energies ([float]): list of predictions for formation
                energies corresponding to candidate_data ordering
            sort (bool): whether or not to sort final list
            floor (float): a float intended to add a floor to the predicted
                formation energies

        Returns:
            (DataFrame): dataframe corresponding to self.candidate_data
        """
        # Preprocess formation energies with floor
        if floor is not None:
            formation_energies = np.array(formation_energies)
            formation_energies[formation_energies < floor] = floor

        # Update formation energy predictions
        self.candidate_data['pred_delta_e'] = formation_energies

        # Construct candidate phases
        candidate_phases = [
            Phase(data['Composition'], energy=data['pred_delta_e'],
                  per_atom=True, description=m_id)
            for m_id, data in self.candidate_data.iterrows()
        ]

        # Refresh and copy seed PD
        pd_ml = deepcopy(self.get_pd())
        pd_ml.add_phases(candidate_phases)
        space_ml = PhaseSpaceAL(bounds=ELEMENTS, data=pd_ml)

        # Compute and return stabilities
        if self.multiprocessing:
            space_ml.compute_stabilities_multi(candidate_phases)
        else:
            space_ml.compute_stabilities_mod(candidate_phases)

        self.candidate_data['pred_stability'] = \
            [phase.stability for phase in candidate_phases]

        if sort:
            self.candidate_data = self.candidate_data.sort_values('pred_stability')

        return self.candidate_data


@camd_traced
class QBCStabilityAgent(StabilityAgent):
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, alpha=0.5, pd=None, multiprocessing=True,
                 ml_algorithm=None, ml_algorithm_params=None, n_members=10):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            ml_algorithm (sklearn-style regressor): Regression method
            ml_algorithm_params (dict): parameters to pass to the regression
                method
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
            n_members (int): number of committee members for the qbc
        """

        super(QBCStabilityAgent, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, hull_distance=hull_distance,
            pd=pd, multiprocessing=multiprocessing
        )

        self.ml_algorithm = ml_algorithm
        self.ml_algorithm_params = ml_algorithm_params
        self.n_members = n_members
        self.alpha = alpha
        self.qbc = QBC(
            N_members=self.n_members, frac=self.frac,
            ML_algorithm=self.ml_algorithm,
            ML_algorithm_params=self.ml_algorithm_params)

    def get_hypotheses(self, candidate_data, seed_data=None,
                       retrain_committee=True):
        self.seed_data = seed_data

        # Retrain committee if untrained or if specified
        if not self.qbc.trained or retrain_committee:
            self.qbc.fit(self.seed_data, self.seed_data['delta_e'],
                         ignore_columns=['Composition', 'N_species', 'delta_e'])
        self.cv_score = self.qbc.cv_score

        # QBC makes predictions for Hf and uncertainty on candidate data
        preds, stds = self.qbc.predict(
            self.candidate_data, ignore_columns=['Composition', 'N_species', 'delta_e'])

        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        self.indices_to_compute = within_hull.head(self.n_query).index
        return self.indices_to_compute


@camd_traced
class AgentStabilityML5(StabilityAgent):
    """
    An agent that does a certain fraction of full exploration and
    exploitation in each iteration.  It will exploit a fraction of
    N_query options (frac), and explore the rest of its budget.
    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True,
                 ml_algorithm=None, ml_algorithm_params=None,
                 exploit_fraction=0.5):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            ml_algorithm (sklearn-style regressor): Regression method
            ml_algorithm_params (dict): parameters to pass to the regression
                method
            exploit_fraction (float): fraction of n_query to assign to
                exploitation hypotheses
        """
        super(AgentStabilityML5, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, pd=pd, hull_distance=hull_distance,
            multiprocessing=multiprocessing
        )

        self.ml_algorithm = ml_algorithm
        self.ml_algorithm_params = ml_algorithm_params
        self.exploit_fraction = exploit_fraction

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.seed_data = seed_data

        X = self.seed_data.drop(['Composition', 'N_species', 'delta_e'], axis=1)
        steps = [('scaler', StandardScaler()), ('ML', self.ml_algorithm(**self.ml_algorithm_params))]
        pipeline = Pipeline(steps)

        cv_score = cross_val_score(pipeline, X, self.seed_data['delta_e'],
                                   cv=KFold(5, shuffle=True), scoring='neg_mean_absolute_error')
        self.cv_score = np.mean(cv_score)*-1
        pipeline.fit(X, self.seed_data['delta_e'])

        # TODO: more general data filtering
        # Dropping columns not relevant for ML predictions, but also
        # 'delta_e' column, if exists. The latter is to ensure delta_e
        # does not end up in features if using after the fact data.
        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        cand_X = self.candidate_data.drop(columns_to_drop, axis=1)
        expected = pipeline.predict(cand_X)

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Filter by stability according to hull distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        # Exploitation part:
        n_exploitation = int(self.n_query * self.exploit_fraction)
        to_compute = list(within_hull.head(n_exploitation).index)
        remaining = within_hull.tail(len(within_hull) - n_exploitation)

        # Exploration part (pick randomly from remainder):
        n_exploration = self.n_query - n_exploitation
        to_compute.extend(remaining.sample(n_exploration).index)

        self.indices_to_compute = to_compute
        return self.indices_to_compute


class GaussianProcessStabilityAgent(StabilityAgent):
    """
    Simple Gaussian Process Regressor based Stability Agent
    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True,
                 alpha=0.5):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
        """
        super(GaussianProcessStabilityAgent, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, pd=pd, hull_distance=hull_distance,
            multiprocessing=multiprocessing
        )
        self.multiprocessing = multiprocessing
        self.alpha = alpha
        self.GP = GaussianProcessRegressor(kernel=ConstantKernel(1) * RBF(1), alpha=0.002)

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.seed_data = seed_data

        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        X = self.seed_data.drop(columns_to_drop, axis=1)
        y = self.seed_data['delta_e']

        steps = [('scaler', StandardScaler()), ('GP', self.GP)]
        cv_pipeline = Pipeline(steps)
        self.cv_score = np.mean(
            -1.0 * cross_val_score(cv_pipeline, X, y, cv=KFold(3, shuffle=True),
                                   scoring='neg_mean_absolute_error'))

        steps = [('scaler', StandardScaler()), ('GP', self.GP)]
        overall_pipeline = Pipeline(steps)
        overall_pipeline.fit(X, y)

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        preds, stds = overall_pipeline.predict(
            self.candidate_data.drop(columns_to_drop, axis=1), return_std=True)
        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        self.indices_to_compute = within_hull.head(self.n_query).index
        return self.indices_to_compute


class SVGProcessStabilityAgent(StabilityAgent):
    """
    Stochastic variational gaussian process stability agent for Big Data.

    The computational complexity of this algorithm scales as O(M^3)
    compared to O(N^3) of standard GP, where N is the number of data points
    and M is the number of inducing points (M<<N).

    The default parameters are optimized to deliver a compromise between
    compute-time and model accuracy for data sets with up to 25 to 40
    thousand examples (e.g. the ICSD seed data). For bigger systems,
    parameter M may need to be reduced. For smaller systems, it can be
    increased, if higher accuracy is desired.  Inducing point locations
    are determined using k-means clustering.

    References:
        Hensman, James, Nicolo Fusi, and Neil D. Lawrence. "Gaussian
            processes for big data." Uncertainty in Artificial
            Intelligence (2013).
        Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic
            optimization." arXiv preprint arXiv:1412.6980 (2014).

    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True,
                 alpha=0.5, M=600):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
        """
        super(SVGProcessStabilityAgent, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, pd=pd, hull_distance=hull_distance,
            multiprocessing=multiprocessing
        )
        self.alpha = alpha if alpha else 0.5
        self.M = M if M else 600

        # Define non-argument SVG-specific attributes
        self.kernel = gpflow.kernels.RBF(273) * gpflow.kernels.Constant(273)
        self.mean_f = gpflow.mean_functions.Constant()
        self.logger = None
        self.model = None
        self.pred_y = None
        self.pred_std = None

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.seed_data = seed_data

        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        X = self.seed_data.drop(columns_to_drop, axis=1)
        y = self.seed_data['delta_e']

        # Test model performance first.  Note we avoid doing CV to
        # reduce compute time. We simply do a 1-way split 80:20 (train:test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Do a k-means clustering and use cluster centers as inducing points.
        cls = MiniBatchKMeans(n_clusters=self.M, batch_size=200)
        cls.fit(X_train_scaled)
        Z = cls.cluster_centers_
        _y = np.array(y_train.to_list())
        mu = np.mean(_y)
        sig = np.std(_y)
        print(Z, _y.shape)
        print(sig, mu)
        model = gpflow.models.SVGP(
            X_train_scaled, ((_y - mu)/sig).reshape(-1, 1), self.kernel,
            gpflow.likelihoods.Gaussian(), Z, mean_function=self.mean_f,
            minibatch_size=100
        )
        print("training")
        t0 = time.time()
        logger = self.run_adam(model, gpflow.test_util.notebook_niter(20000))
        print("elapsed time: ", time.time()-t0)

        pred_y, pred_v = model.predict_y(scaler.transform(X_test))
        pred_y = pred_y * sig + mu
        self.cv_score = np.mean(np.abs(pred_y - y_test.to_numpy().reshape(-1, 1)))
        print("cv score", self.cv_score)
        self.model = model

        # Overall model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cls = MiniBatchKMeans(n_clusters=self.M, batch_size=200)
        cls.fit(X_train_scaled)
        Z = cls.cluster_centers_
        _y = np.array(y.to_list())
        mu = np.mean(_y)
        sig = np.std(_y)
        model = gpflow.models.SVGP(
            X_scaled, ((_y - mu)/sig).reshape(-1, 1), self.kernel,
            gpflow.likelihoods.Gaussian(), Z, mean_function=self.mean_f,
            minibatch_size=100
        )
        logger = self.run_adam(model, gpflow.test_util.notebook_niter(20000))
        print(self.model)
        self.model = model

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        pred_y, pred_v =  model.predict_y(scaler.transform( self.candidate_data.drop(columns_to_drop, axis=1)))
        pred_y = pred_y * sig + mu
        self.pred_y = pred_y.reshape(-1,)
        self.pred_std = (pred_v**0.5).reshape(-1,)

        expected = self.pred_y - self.pred_std * self.alpha
        print("expected improv", expected)

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        self.indices_to_compute = within_hull.head(self.n_query).index
        return self.indices_to_compute

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


class BaggedGaussianProcessStabilityAgent(StabilityAgent):
    """
    An ensemble GP learner that can handle relatively large
    datasets by bagging. WIP.

    Current strategy is that weak GP learners are trained on
    random subsets of data with max 500 points (and no bootstrapping),
    to learn a stronger, ensemble learner that minimizes model variance.

    Learned model is usually stronger, and on par with Random Forest
    or NN. We assume the uncertainty of the prediction to be
    the minimum std among all GP estimators trained, for a given point.

    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True,
                 alpha=0.5, n_estimators=8, max_samples=5000,
                 bootstrap=False):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            alpha (float): scaling for stdev for estimation of best-
                case formation energies
        """
        super(BaggedGaussianProcessStabilityAgent, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, pd=pd, hull_distance=hull_distance,
            multiprocessing=multiprocessing
        )

        self.alpha = alpha
        self.n_estimators = n_estimators if n_estimators else 8
        self.max_samples = max_samples if max_samples else 5000
        self.bootstrap = bootstrap if bootstrap else False

        self.GP = GaussianProcessRegressor(kernel=ConstantKernel(1) * RBF(1), alpha=0.002)

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.seed_data = seed_data

        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        X = self.seed_data.drop(columns_to_drop, axis=1)
        y = self.seed_data['delta_e']

        steps = [('scaler', StandardScaler()), ('GP', self.GP)]
        pipeline = Pipeline(steps)

        bag_reg = BaggingRegressor(
            base_estimator=pipeline, n_estimators=self.n_estimators,
            max_samples=self.max_samples, bootstrap=self.bootstrap,
            verbose=True, n_jobs=self.n_jobs)
        self.cv_score = np.mean(
            -1.0 * cross_val_score(
                pipeline, X, y, cv=KFold(3, shuffle=True),
                scoring='neg_mean_absolute_error')
        )
        bag_reg.fit(X, y)

        # TODO: this should probably be an static method
        def _get_unc(bagging_regressor, X_test):
            stds = []
            pres = []
            for est in bagging_regressor.estimators_:
                _p, _s = est.predict(X_test, return_std=True)
                stds.append(_s)
                pres.append(_p)
            return np.mean(np.array(pres), axis=0), np.min(np.array(stds), axis=0)

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        preds, stds = _get_unc(bag_reg, self.candidate_data.drop(columns_to_drop, axis=1))
        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        self.indices_to_compute = within_hull.head(self.n_query).index
        return self.indices_to_compute


@camd_traced
class AgentStabilityAdaBoost(StabilityAgent):
    """
    An agent that does a certain fraction of full exploration and
    exploitation in each iteration.  It will exploit a fraction
    of N_query options (frac), and explore the rest of its budget.
    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1,
                 hull_distance=0.0, pd=None, multiprocessing=True,
                 ml_algorithm=None, ml_algorithm_params=None,
                 uncertainty=True, alpha=0.5, n_estimators=10,
                 exploit_fraction=0.5):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            # TODO: is pd necessary as a constructor argument?
            pd (PhaseDiagram): phase diagram to initialize the agent with
            multiprocessing (bool): whether to use multiprocessing
                for phase stability analysis
            ml_algorithm (sklearn-style regressor): Regression method
            ml_algorithm_params (dict): parameters to pass to the regression
                method
            uncertainty (bool): whether uncertainty is included in
                minimal predictions
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
            n_estimators (int): number of estimators fro the AdaBoosting
                algorithm
            exploit_fraction (float): fraction of n_query to assign to
                exploitation hypotheses
        """

        super(AgentStabilityAdaBoost, self).__init__(
            candidate_data=candidate_data, seed_data=seed_data,
            n_query=n_query, hull_distance=hull_distance,
            pd=pd, multiprocessing=multiprocessing
        )

        self.ml_algorithm = ml_algorithm
        self.ml_algorithm_params = ml_algorithm_params
        self.frac = exploit_fraction
        self.uncertainty = uncertainty
        self.alpha = alpha
        self.n_estimators = n_estimators

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.seed_data = seed_data

        X = self.seed_data.drop(['Composition', 'N_species', 'delta_e'], axis=1)
        steps = [('scaler', StandardScaler()), ('ML', self.ml_algorithm(**self.ml_algorithm_params))]
        pipeline = Pipeline(steps)

        adaboost = AdaBoostRegressor(
            base_estimator=pipeline, n_estimators=self.n_estimators)

        cv_score = cross_val_score(
            adaboost, X, self.seed_data['delta_e'],
            cv=KFold(3, shuffle=True), scoring='neg_mean_absolute_error')
        self.cv_score = np.mean(cv_score)*-1

        # We will take standard scaler out of the pipleine for
        # prediction purposes (we want a single scaler)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        overall_adaboost = AdaBoostRegressor(
            base_estimator=self.ml_algorithm(**self.ml_algorithm_params),
            n_estimators=self.n_estimators
        )
        overall_adaboost.fit(X_scaled, self.seed_data['delta_e'])

        # Dropping columns not relevant for ML predictions, but also
        # 'delta_e' column, if exists. The latter is to ensure delta_e does not end up in features if using
        # after the fact data.
        columns_to_drop = ['Composition', 'N_species', 'delta_e']
        cand_X = self.candidate_data.drop(columns_to_drop, axis=1)

        cand_X = scaler.transform(cand_X)
        expected = overall_adaboost.predict(cand_X)

        if self.uncertainty:
            expected -= self.alpha * self._get_unc_ada(overall_adaboost, cand_X)

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(
            expected, sort=True, floor=-6.0)

        # Filter by stability according to hull distance
        stability_filter = self.candidate_data['pred_stability'] < self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        # Exploitation part:
        n_exploitation = int(self.n_query * self.exploit_fraction)
        to_compute = list(within_hull.head(n_exploitation).index)
        remaining = within_hull.tail(len(within_hull) - n_exploitation)

        # Exploration part (pick randomly from remainder):
        n_exploration = self.n_query - n_exploitation
        to_compute.extend(remaining.sample(n_exploration).index)

        self.indices_to_compute = to_compute
        return self.indices_to_compute

    @staticmethod
    def _get_unc_ada(ada, X):
        preds = []
        for i in ada.estimators_:
            preds.append(i.predict(X))
        preds = np.array(preds)
        preds = preds.T
        stds = []
        for i in preds:
            average = np.average(i, weights=ada.estimator_weights_)
            _std = np.sqrt(np.average((i - average) ** 2, weights=ada.estimator_weights_))
            stds.append(_std)
        return np.array(stds)
