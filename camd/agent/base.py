# Copyright Toyota Research Institute 2019
"""
Module containing basic agent Abstractions
and functionality
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone

from camd import tqdm

import abc


class HypothesisAgent(metaclass=abc.ABCMeta):
    """
    Abstract class for agents, decision-making entities
    in a sequential learning setting.  Should implement
    a `get_hypotheses` method that takes a list of potential
    candidates and selects those which are most well-suited
    to experiments
    """
    def __init__(self):
        """
        Placeholder for initializing agents, should include
        all state parameters that an agent needs to make
        decisions
        """
        pass

    @abc.abstractmethod
    def get_hypotheses(self, candidate_data, seed_data=None):
        """

        Returns:
            subset of candidate data which represent some
            choice e. g. for the next set of experiments

        """


class QBC:
    """
    Helper class for Uncertainty quantification using
    non-supporting regressors with Query-By-Committee
    """

    def __init__(self, n_members, training_fraction, model=None, test_full_model=True):
        """
        Args:
            n_members (int): Number of committee members or models to train
            training_fraction (float): fraction of data to use in training
                committee members
            model (sklearn.RegressorMixin): sklearn-style regressor
            test_full_model (bool): whether or not to test the full
                model

        """
        self.n_members = n_members
        self.training_fraction = training_fraction
        self.model = model if model else LinearRegression()
        self.committee_models = []
        self.trained = False
        self.test_full_model = test_full_model
        self.cv_score = np.nan
        self._X = None
        self._y = None

    def fit(self, X, y):
        """
        Fits the QBC committee member models

        Args:
            X (pandas.DataFrame, np.ndarray): input X values for fitting
            y (pandas.DataFrame, np.ndarray): output y values to regress
                or fit to

        Returns:
            None

        """
        self._X, self._y = X, y

        split_X = []
        split_y = []

        for i in range(self.n_members):
            a = np.arange(len(X))
            np.random.shuffle(a)
            indices = a[: int(self.training_fraction * len(X))]
            split_X.append(X.iloc[indices])
            split_y.append(y.iloc[indices])

        self.committee_models = []
        for i in tqdm(list(range(self.n_members))):
            scaler = StandardScaler()
            X = scaler.fit_transform(split_X[i])
            y = split_y[i]
            model = clone(self.model)
            model.fit(X, y)
            # Saving the scaler and model to make predictions
            self.committee_models.append([scaler, model])

        self.trained = True

        if self.test_full_model:
            # Get a CV score for an overall model with plot_hull dataset
            full_scaler = StandardScaler()
            _X = full_scaler.fit_transform(self._X, self._y)
            full_model = clone(self.model)
            full_model.fit(_X, self._y)
            cv_score = cross_val_score(
                full_model,
                _X,
                self._y,
                cv=KFold(5, shuffle=True),
                scoring="neg_mean_absolute_error",
            )
            self.cv_score = np.mean(cv_score) * -1

    def predict(self, X):
        """
        Apply the fitted committee of models to candidate space

        Args:
            X (pandas.DataFrame, np.ndarray): input matrix or values
                on which to predict

        Returns:
            (np.ndarray): mean values for predictions for all committee members
            (np.ndarray): standard deviation values for predictions for all committee members

        """
        committee_predictions = []
        for scaler, model in tqdm(self.committee_models):
            _X = scaler.transform(X)
            committee_predictions.append(model.predict(_X))
        stds = np.std(np.array(committee_predictions), axis=0)
        means = np.mean(np.array(committee_predictions), axis=0)
        return means, stds


class RandomAgent(HypothesisAgent):
    """
    Baseline agent: Randomly picks from candidate dataset
    """

    def __init__(self, candidate_data=None, seed_data=None, n_query=1):
        """
        Initializes a random agent

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): seed data, in this case does nothing
            n_query (int): number of candidates to query

        """

        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        super(RandomAgent, self).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        """

        Args:
            candidate_data (DataFrame): candidate data
            seed_data (DataFrame): seed data, there's none in this
                case, but keep the kwarg for adherence to the
                superclass signature

        Returns:
            (List) of indices

        """
        return candidate_data.sample(self.n_query)
