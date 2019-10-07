# Copyright Toyota Research Institute 2019

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from camd import tqdm

import abc


class HypothesisAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_hypotheses(self, candidate_data):
        """

        Returns:
            subset of candidate data which represent some
            choice e. g. for the next set of experiments

        """


class QBC:
    """
    Uncertainty quantification for non-supporting regressors with Query-By-Committee
    """
    def __init__(self, N_members, frac, ML_algorithm=None, ML_algorithm_params=None,
                 test_full_model=True):
        """
        :param N_members: Number of committee members (i.e. models to train)
        :param frac: fraction of data to use in training committee members
        :param ML_algorithm: sklearn-style regressor
        :param ML_algorithm_params: (dict) parameters to pass to the algorithm
        """
        self.N_members = N_members
        self.frac = frac
        self.ML_algorithm = ML_algorithm if ML_algorithm else LinearRegression
        self.ML_algorithm_params = ML_algorithm_params if ML_algorithm_params else {}
        self.committee_models = []
        self.ignore_columns = None
        self.trained = False
        self.test_full_model = test_full_model
        self.cv_score = np.nan

    def fit(self, X, y, ignore_columns=None):

        self.ignore_columns = ignore_columns if ignore_columns else []
        self._X = X.drop(ignore_columns, axis=1)
        self._y = y

        split_X = []
        split_y = []

        for i in range(self.N_members):
            a = np.arange(len(X))
            np.random.shuffle(a)
            indices = a[:int(self.frac * len(X))]
            split_X.append(X.iloc[indices])
            split_y.append(y.iloc[indices])

        self.committee_models = []
        for i in tqdm(list(range(self.N_members))):
            scaler = StandardScaler()
            X = scaler.fit_transform(split_X[i].drop(ignore_columns, axis=1))
            y = split_y[i]
            model = self.ML_algorithm(**self.ML_algorithm_params)
            model.fit(X, y)
            self.committee_models.append([scaler, model])  # Note we're saving the scaler to use in predictions

        self.trained = True

        if self.test_full_model:
            # Get a CV score for an overall model with present dataset
            overall_model = self.ML_algorithm(**self.ML_algorithm_params)
            overall_scaler = StandardScaler()
            _X = overall_scaler.fit_transform(self._X, self._y)
            overall_model.fit(_X, self._y)
            cv_score = cross_val_score(overall_model, _X, self._y,
                                       cv=KFold(5, shuffle=True), scoring='neg_mean_absolute_error')
            self.cv_score = np.mean(cv_score) * -1

    def predict(self, X, ignore_columns=None):

        ignore_columns = ignore_columns if ignore_columns else self.ignore_columns

        # Apply the committee of models to candidate space
        committee_predictions = []
        for i in tqdm(list(range(self.N_members))):
            scaler = self.committee_models[i][0]
            model = self.committee_models[i][1]
            _X = X.drop(ignore_columns, axis=1)
            _X = scaler.transform(_X)
            committee_predictions.append(model.predict(_X))
        stds = np.std(np.array(committee_predictions), axis=0)
        means = np.mean(np.array(committee_predictions), axis=0)

        return means, stds


class RandomAgent(HypothesisAgent):
    """
    Baseline agent: Randomly picks next experiments
    """
    def __init__(self, candidate_data=None, seed_data=None, n_query=1):

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        self.cv_score = np.nan
        super(RandomAgent, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        """

        Args:
            candidate_data (DataFrame): candidate data
            seed_data (DataFrame): seed data

        Returns:
            (List) of indices

        """
        return self.candidate_data.sample(self.n_query).index.tolist()
