# Copyright Toyota Research Institute 2019
"""
Contains agents related to crystal-structure discovery
and hypothesizing
"""

import abc
import json
import os
from copy import deepcopy
from multiprocessing import cpu_count
from collections import OrderedDict

import numpy as np
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from camd.analysis import PhaseSpaceAL, ELEMENTS
from camd.agent.base import HypothesisAgent, QBC
from camd.utils.data import filter_dataframe_by_composition
from pymatgen.core.composition import Composition

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import pairwise_distances

# TODO: Adaptive N_query and subsampling of candidate space
# TODO: Exploit/Explore general method
# TODO: General uncertainty handling?


class StabilityAgent(HypothesisAgent, metaclass=abc.ABCMeta):
    """
    The StabilityAgent is a mixin abstract class which contains
    initialization parameters and methods common to every agent
    which is responsible for making decisions about stability.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool, int): whether to use multiprocessing
                for phase stability analysis, if an int, sets the n_jobs
                parameter as well.  If a bool, sets n_jobs to cpu_count()
                if True and n_jobs to 1 if false.
        """
        super().__init__()
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        self.hull_distance = hull_distance
        self.pd = None
        self.parallel = parallel

        # These might be able to go into the base class
        self.cv_score = np.nan

    def get_pd(self, chemsys=None):
        """
        Refresh the phase diagram associated with the seed_data

        Args:
            chemsys (str): chemical system for which to filter
                seed data to provide partial phase diagram

        Returns:
            None
        """
        self.pd = PhaseData()
        # Filter seed data by relevant chemsys
        if chemsys:
            total_comp = Composition(chemsys.replace('-', ''))
            filtered = filter_dataframe_by_composition(
                self.seed_data, total_comp)
        else:
            filtered = self.seed_data

        phases = [
            Phase(
                row["Composition"],
                energy=row["delta_e"],
                per_atom=True,
                description=row_index,
            )
            for row_index, row in filtered.iterrows()
        ]
        phases.extend([Phase(el, 0.0, per_atom=True) for el in ELEMENTS])
        self.pd.add_phases(phases)
        return self.pd

    def update_data(self, candidate_data=None, seed_data=None):
        """
        Helper function to update the data according to the schema
        of the default OQMD data.  Updates the candidate_data and
        seed_data attributes, and returns the processed features
        and targets associated with the candidates and seed data.

        Args:
            candidate_data (DataFrame): new candidate dataframe
            seed_data (DataFrame): new seed dataframe

        Returns:
            (DataFrame): candidate features
            (DataFrame): seed features
            (DataFrame): seed targets

        """
        # Note: In the drop command, we're ignoring errors for
        #   brevity.  We should watch this, because we may not
        #   drop everything we intend to.
        drop_columns = [
            "Composition",
            "N_species",
            "delta_e",
            "pred_delta_e",
            "pred_stability",
            "stability",
            "is_stable",
            "structure",
        ]
        if candidate_data is not None:
            self.candidate_data = candidate_data
            X_cand = candidate_data.drop(drop_columns, axis=1, errors="ignore")
        else:
            X_cand = None
        if seed_data is not None:
            self.seed_data = seed_data
            X_seed = self.seed_data.drop(drop_columns, axis=1, errors="ignore")
            y_seed = self.seed_data["delta_e"]
        else:
            X_seed, y_seed = None, None

        return X_cand, X_seed, y_seed

    def update_candidate_stabilities(self, formation_energies, sort=True, floor=-6.0):
        """
        Updates the candidate dataframe with the stabilities
        of the candidate compositions according to the requisite
        phase diagram analysis.

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
        self.candidate_data["pred_delta_e"] = formation_energies

        # Construct candidate phases
        candidate_phases = [
            Phase(
                data["Composition"],
                energy=data["pred_delta_e"],
                per_atom=True,
                description=m_id,
            )
            for m_id, data in self.candidate_data.iterrows()
        ]

        # Refresh and copy seed PD filtered by candidates
        all_comp = self.candidate_data['Composition'].sum()
        pd_ml = deepcopy(self.get_pd(all_comp))
        pd_ml.add_phases(candidate_phases)
        space_ml = PhaseSpaceAL(bounds=ELEMENTS, data=pd_ml)

        # Compute and return stabilities
        space_ml.compute_stabilities(candidate_phases, self.parallel)
        self.candidate_data["pred_stability"] = [
            phase.stability for phase in candidate_phases
        ]

        if sort:
            self.candidate_data = self.candidate_data.sort_values("pred_stability")

        return self.candidate_data


class QBCStabilityAgent(StabilityAgent):
    """
    Agent which uses QBC to determine optimal hypotheses
    """
    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
        alpha=0.5,
        training_fraction=0.5,
        model=None,
        n_members=10,
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool): whether to use multiprocessing
                for phase stability analysis
            training_fraction (float): fraction of data to use for
                training committee members
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
            model (sklearn-style regressor): regressor
            n_members (int): number of committee members for the qbc
        """

        super(QBCStabilityAgent, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )

        self.alpha = alpha
        self.model = model
        self.n_members = n_members
        self.qbc = QBC(
            n_members=n_members, training_fraction=training_fraction, model=model,
        )

    def get_hypotheses(self, candidate_data, seed_data=None, retrain_committee=True):
        """
        Get hypotheses method for QBCStabilityAgent

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit GPUCB
            retrain_committee (bool): whether to retrain committee
                each time

        Returns:
            (pandas.DataFrame): top candidates from the GPUCB algorithm

        """
        X_cand, X_seed, y_seed = self.update_data(candidate_data, seed_data)

        # Retrain committee if untrained or if specified
        if not self.qbc.trained or retrain_committee:
            self.qbc.fit(X_seed, y_seed)
        self.cv_score = self.qbc.cv_score

        # QBC makes predictions for Hf and uncertainty on candidate data
        preds, stds = self.qbc.predict(X_cand)
        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        return within_hull.head(self.n_query)


class AgentStabilityML5(StabilityAgent):
    """
    An agent that does a certain fraction of full exploration and
    exploitation in each iteration.  It will exploit a fraction of
    N_query options (frac), and explore the rest of its budget.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
        model=None,
        exploit_fraction=0.5,
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool): whether to use multiprocessing
                for phase stability analysis
            model (sklearn-style Regressor): Regression method
            exploit_fraction (float): fraction of n_query to assign to
                exploitation hypotheses
        """
        super(AgentStabilityML5, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )

        self.model = model or LinearRegression()
        self.exploit_fraction = exploit_fraction

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Get hypotheses method for AgentStabilityML5

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit GPUCB

        Returns:
            (pandas.DataFrame): top candidates from the GPUCB algorithm

        """

        X_cand, X_seed, y_seed = self.update_data(candidate_data, seed_data)
        steps = [("scaler", StandardScaler()), ("ML", self.model)]
        pipeline = Pipeline(steps)

        cv_score = cross_val_score(
            pipeline,
            X_seed,
            self.seed_data["delta_e"],
            cv=KFold(5, shuffle=True),
            scoring="neg_mean_absolute_error",
        )
        self.cv_score = np.mean(cv_score) * -1
        pipeline.fit(X_seed, self.seed_data["delta_e"])

        expected = pipeline.predict(X_cand)

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

        # Filter by stability according to hull distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        # Exploitation part:
        n_exploitation = int(self.n_query * self.exploit_fraction)
        to_compute = within_hull.head(n_exploitation).index.tolist()
        remaining = within_hull.tail(len(within_hull) - n_exploitation)
        remaining = remaining.append(self.candidate_data[~stability_filter])

        # Exploration part (pick randomly from remainder):
        n_exploration = self.n_query - n_exploitation

        if n_exploration > len(remaining):
            n_exploration = len(remaining)
        to_compute.extend(remaining.sample(n_exploration).index.tolist())
        return candidate_data.loc[to_compute]


class GaussianProcessStabilityAgent(StabilityAgent):
    """
    Simple Gaussian Process Regressor based Stability Agent
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
        alpha=0.5,
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool): whether to use multiprocessing
                for phase stability analysis
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
        """
        super(GaussianProcessStabilityAgent, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )
        self.multiprocessing = parallel
        self.alpha = alpha
        self.GP = GaussianProcessRegressor(
            kernel=ConstantKernel(1) * RBF(1), alpha=0.002
        )

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Get hypotheses method for GaussianProcessStabilityAgent

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit GPUCB

        Returns:
            (pandas.DataFrame): top candidates from the GPUCB algorithm

        """

        X_cand, X_seed, y_seed = self.update_data(
            candidate_data=candidate_data, seed_data=seed_data
        )

        steps = [("scaler", StandardScaler()), ("GP", self.GP)]
        cv_pipeline = Pipeline(steps)
        self.cv_score = np.mean(
            -1.0
            * cross_val_score(
                cv_pipeline,
                X_seed,
                y_seed,
                cv=KFold(3, shuffle=True),
                scoring="neg_mean_absolute_error",
            )
        )

        steps = [("scaler", StandardScaler()), ("GP", self.GP)]
        overall_pipeline = Pipeline(steps)
        overall_pipeline.fit(X_seed, y_seed)

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        preds, stds = overall_pipeline.predict(X_cand, return_std=True)
        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        return within_hull.head(self.n_query)


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

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
        alpha=0.5,
        n_estimators=8,
        max_samples=5000,
        bootstrap=False,
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool): whether to use multiprocessing
                for phase stability analysis
            alpha (float): scaling for stdev for estimation of best-
                case formation energies
        """
        super(BaggedGaussianProcessStabilityAgent, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )

        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap

        self.GP = GaussianProcessRegressor(
            kernel=ConstantKernel(1) * RBF(1), alpha=0.002
        )

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Get hypotheses method for BaggedGaussianProcessStabilityAgent

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit model

        Returns:
            (pandas.DataFrame): top candidates from the algorithm

        """

        X_cand, X_seed, y_seed = self.update_data(candidate_data, seed_data)

        steps = [("scaler", StandardScaler()), ("GP", self.GP)]
        pipeline = Pipeline(steps)
        n_jobs = self.parallel if self.parallel else None
        bag_reg = BaggingRegressor(
            base_estimator=pipeline,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
            verbose=True,
            n_jobs=n_jobs,
        )
        self.cv_score = np.mean(
            -1.0
            * cross_val_score(
                pipeline,
                X_seed,
                y_seed,
                cv=KFold(3, shuffle=True),
                scoring="neg_mean_absolute_error",
            )
        )
        bag_reg.fit(X_seed, y_seed)

        # GP makes predictions for Hf and uncertainty*alpha on candidate data
        preds, stds = self._get_unc(bag_reg, X_cand)
        expected = preds - stds * self.alpha

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]
        return within_hull.head(self.n_query)

    @staticmethod
    def _get_unc(bagging_regressor, X_test):
        """

        Args:
            bagging_regressor (RegressorMixin): regressor for which
                to get uncertainty
            X_test (np.ndarray): test data on which to estimate
                uncertainty

        Returns:
            (np.ndarray): array of uncertainty values

        """
        stds = []
        pres = []
        for est in bagging_regressor.estimators_:
            _p, _s = est.predict(X_test, return_std=True)
            stds.append(_s)
            pres.append(_p)
        return np.mean(np.array(pres), axis=0), np.min(np.array(stds), axis=0)


class AgentStabilityAdaBoost(StabilityAgent):
    """
    An agent that does a certain fraction of full exploration and
    exploitation in each iteration.  It will exploit a fraction
    of N_query options (frac), and explore the rest of its budget.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
        model=None,
        uncertainty=True,
        alpha=0.5,
        n_estimators=10,
        exploit_fraction=0.5,
        diversify=False,
        dynamic_alpha=False,
    ):
        """
        Args:
            candidate_data (DataFrame): data about the candidates
            seed_data (DataFrame): data which to fit the Agent to
            n_query (int): number of hypotheses to generate
            hull_distance (float): hull distance as a criteria for
                which to deem a given material as "stable"
            parallel (bool): whether to use multiprocessing
                for phase stability analysis
            model (sklearn-style regressor): Regression method
            uncertainty (bool): whether uncertainty is included in
                minimal predictions
            alpha (float): weighting factor for the stdev in making
                best-case predictions of the stability
            n_estimators (int): number of estimators for the AdaBoosting
                algorithm
            exploit_fraction (float): fraction of n_query to assign to
                exploitation hypotheses
            diversify (bool): Turns on the diversification algorithm for
                selected experiments.
            dynamic_alpha (bool): Turns on a simple linear schedule where the
                alpha (how much uncertainty is mixed into predictions) starts
                from zero, and increases at a rate of 0.1/iter until it's capped
                at parameter alpha specified for the rest of the iterations.
        """

        super(AgentStabilityAdaBoost, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )
        self.model = model
        self.exploit_fraction = exploit_fraction
        self.uncertainty = uncertainty
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.diversify = diversify
        self.dynamic_alpha = dynamic_alpha

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Get hypotheses method for AgentStabilityAdaBoost

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit model

        Returns:
            (pandas.DataFrame): top candidates from the algorithm

        """
        X_cand, X_seed, y_seed = self.update_data(candidate_data, seed_data)

        steps = [("scaler", StandardScaler()), ("ML", self.model)]
        pipeline = Pipeline(steps)

        adaboost = AdaBoostRegressor(
            base_estimator=pipeline, n_estimators=self.n_estimators
        )

        cv_score = cross_val_score(
            adaboost,
            X_seed,
            y_seed,
            cv=KFold(3, shuffle=True),
            scoring="neg_mean_absolute_error",
        )
        self.cv_score = np.mean(cv_score) * -1

        # We will take standard scaler out of the pipleine for
        # prediction purposes (we want a single scaler)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_seed)
        overall_adaboost = AdaBoostRegressor(
            base_estimator=self.model, n_estimators=self.n_estimators
        )
        overall_adaboost.fit(X_scaled, self.seed_data["delta_e"])

        X_cand = scaler.transform(X_cand)
        expected = overall_adaboost.predict(X_cand)

        if self.uncertainty:
            if self.dynamic_alpha and os.path.exists("iteration.json"):
                with open("iteration.json", "r") as f:
                    iter = json.load(f)
                print("dynamic alpha activated")
                expected -= min(0.1 * iter, self.alpha) * self._get_unc_ada(
                    overall_adaboost, X_cand
                )
            else:
                expected -= self.alpha * self._get_unc_ada(overall_adaboost, X_cand)

        # Update candidate data dataframe with predictions
        self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

        # Filter by stability according to hull distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]

        # Exploitation part:
        n_exploitation = int(self.n_query * self.exploit_fraction)
        if self.diversify:
            to_compute = diverse_quant(
                within_hull.index.tolist(), n_exploitation, self.candidate_data
            )
        else:
            to_compute = within_hull.head(n_exploitation).index.tolist()
        remaining = within_hull.tail(len(within_hull) - n_exploitation)
        remaining = remaining.append(self.candidate_data[~stability_filter])

        # Exploration part (pick randomly from remainder):
        n_exploration = self.n_query - n_exploitation

        if n_exploration > len(remaining):
            n_exploration = len(remaining)
        to_compute.extend(remaining.sample(n_exploration).index.tolist())

        return candidate_data.loc[to_compute]

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
            _std = np.sqrt(
                np.average((i - average) ** 2, weights=ada.estimator_weights_)
            )
            stds.append(_std)
        return np.array(stds)


def diverse_quant(points, target_length, df, quantiles=None):
    """
    Diversify a sublist by eliminating entries based on comparisons
    with quantiles threshold and Euclidean distance.

    This method takes the points list (which would be the object
    within_hull in stability implementations), and selects a diverse
    subset for the number of exploitation choices (target_length).

    It follows a simple algorithm: start from i = 0 of the points,
    and go down the remainder of the list, removing entries that seem
    to be closer to i below a certain distance threshold. It repeats
    this process for all i: i = 1, i = 2 ... i_max.

    The method tries to adjust the distance threshold until it finds
    the shortest resulting list that is longer than target_length. If
    it can't, it will return points as it is. The threshold values are
    decided by finding distances corresponding to quantiles of the a
    sampled distribution of distances in the overall feature set.

    The method does not alter the original ordering in the list points.

    The intuition behind the algorithm is to make risk-averse choices
    by avoiding the acquisition of too similar candidates, in case one
    example among those entries is sufficient for the model to minimize
    its uncertainty and/or make a decision to not acquire any other in
    that region.

    So the resources would not be wasted and can be allocated to other
    promising choices in points.

    Args:
        points (list): Initial set of points reflecting a preferred order
        target_length (int): length of desired sublist
        df (DataFrame): feature vectors of points, where index labels
            contain elements of points
        quantiles (list): quantilies to test for threshold.
            Defaults to [0.01, 0.02, 0.03, 0.04, 0.05]

    Returns:
        A diversified sublist of points

    """
    quantiles = quantiles if quantiles else [0.01, 0.02, 0.03, 0.04, 0.05]
    if target_length >= len(points):
        return points
    if len(df) > 6000:
        _df = df.sample(6000)
    else:
        _df = df
    drop_columns = [
        "Composition",
        "N_species",
        "delta_e",
        "pred_delta_e",
        "pred_stability",
        "structure"
    ]
    scaler = StandardScaler()
    X = scaler.fit_transform(_df.drop(drop_columns, axis=1, errors="ignore"))
    flatD = pairwise_distances(X).flatten()

    _df2 = df.loc[points]
    X = scaler.transform(_df2.drop(drop_columns, axis=1, errors="ignore"))
    D = pairwise_distances(X)

    remove_len = len(points) - target_length
    res = []
    for alpha in quantiles:
        q = np.quantile(flatD, alpha)
        to_remove = []
        for i in range(0, len(points)):
            for j in range(i + 1, len(points)):
                if D[i, j] < q:
                    to_remove.append(j)
        _rl = len(set(to_remove))
        print(_rl, remove_len)
        if _rl >= remove_len:
            break
        res.append(to_remove)
    if len(res) == 0:
        return points[:target_length]
    else:
        d = OrderedDict()
        for i in res[-1]:  # fall back to the latest remove list before break
            d[i] = None
        final_remove_list = list(d.keys())

        return [
            p
            for p in points
            if np.where(_df2.index == p)[0][0] not in final_remove_list
        ][:target_length]
