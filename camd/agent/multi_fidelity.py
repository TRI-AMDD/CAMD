# Copyright Toyota Research Institute 2021
"""
This module implements agents and helper functions designed
support multi-fidelity discovery campaigns.

See the following reference:

Palizhati A, Aykol M, Suram S, Hummelshøj JS, Montoya JH. Multi-fidelity Sequential Learning for
Accelerated Materials Discovery. ChemRxiv. Cambridge: Cambridge Open Engage; 2021; This content
is a preprint and has not been peer-reviewed.
"""


import GPy
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from camd.agent.base import HypothesisAgent


def get_features_from_df(df, features):
    """
    Helper function to get feature columns of a dataframe.

    Args:
        df (pd.DataFrame)       A pd.DataFrame where the features are extracted.
        features (list of str)  Name of the features columns in df.

    Returns:
            feature_df      A pd.DataFrame that only contains the features used in ML.
    """
    feature_df = df[features]
    return feature_df


def process_data(candidate_data, seed_data, features, label, y_reshape=False, preprocessor=None):
    """
    Helper function that process data for ML model and returns
    np.ndarray of training features, training labels,
    test features and test labels.
    """
    X_train = get_features_from_df(seed_data, features).values
    y_train = np.array(seed_data[label])
    X_test = get_features_from_df(candidate_data, features).values
    y_test = np.array(candidate_data[label])

    if y_reshape:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    if preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    return X_train, y_train, X_test, y_test


class EpsilonGreedyMultiAgent(HypothesisAgent):
    """
    A multi-fidelity agent that allocates the desired budget for high fidelity versus
    low fidelity candidate data, and acquires candidates in each fidelity via exploitation
    """
    def __init__(self, candidate_data=None, seed_data=None, features=None, fidelities=('theory_data', 'expt_data'),
                 target_prop=None, target_prop_val=None, preprocessor=StandardScaler(), model=None,
                 ranking_method='minimize', total_budget=None, highFi_query_frac=None, similarity_thres=300.,
                 lowFi_per_highFi=1):
        """
        Args:
            candidate_data (df)      Candidate search space for the campaign.
            seed_data (df)           Seed (training) data for the campaign.
            features (tuple of str)  Name of the feature columns used in machine learning model training.
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged
                                     from low to high fidelity.
            target_prop (str)        The property machine learning model is learning, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            preprocessor             A preprocessor that preprocess the features. It can be None, a single
                                     processor, or a pipeline. The default is sklearn StandardScaler.
            model                    A sklearn supervised machine learning regressor.
            ranking_method (str)     Ranking method of candidates based on ML predictions. Select one of the
                                     following: "minimize", "ascending", or "descending". "minimize" will rank
                                     candidates with ML candidates closest to the target property value. "ascending"
                                     or "descening" will rank candidates with ascending/descening ML predictions, best
                                     to use when there is no target propety value (i.e. smaller/larger the better).
            total_budget (int)       The number of the hypotheses at a given iteration of the campaign.
            highFi_query_frac        The fraction of the total budget used for high fidelity hypotheses queries.
                                     The value should be <0 and <=1.
            similarity_thres(float)  The threshold value for l2 norm similarity between a candidate composition and
                                     compositions in the seed data. User will need to run some calculations
                                     to determine the best threshold value.
            lowFi_per_highFi (int)   The number of low fidelity candidate selected to support each
                                     high fidelity candidates that predicted to be good, but the agent
                                     does not want to generate that experimental hypotheses yet.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.preprocessor = preprocessor
        self.model = model
        self.ranking_method = ranking_method
        self.total_budget = total_budget
        self.highFi_query_frac = highFi_query_frac
        self.similarity_thres = similarity_thres
        self.lowFi_per_highFi = lowFi_per_highFi
        super(EpsilonGreedyMultiAgent).__init__()

    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between
        a composition and the seed data compositions. The similarity
        is reprsented by l2_norm.

        Args:
            comp(pd.core.series):    A specific composition represented by Magpie.
            seed_comps (df):         Compostions in seed represented by Magpie.
        """
        l2_norm = np.linalg.norm(comp.values - seed_comps.values,  axis=1)
        return l2_norm

    def _query_hypotheses(self, candidate_data, seed_data):
        """
        Query hypotheses given candidate and seed data via exploitation.
        """
        # separate the candidate space into high and low fidelity candidates
        high_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[1]] == 1]
        low_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[0]] == 1]

        # edge cases: end campaign if there are no high fidelity candidate
        # use the entire query budget if there are only high fidelity candidate
        if len(high_fidelity_candidates) == 0:
            return None

        elif (len(high_fidelity_candidates) != 0) & (len(low_fidelity_candidates) == 0):
            selected_hypotheses = high_fidelity_candidates.head(self.total_budget)

        else:
            selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)

            # Query high fidelity candidates first
            highFi_budget = int(self.total_budget * self.highFi_query_frac)
            seed_data_fea = get_features_from_df(seed_data, self.features)
            highFi_cands_fea = get_features_from_df(high_fidelity_candidates, self.features)

            for idx, cand_fea in highFi_cands_fea.iterrows():
                if len(selected_hypotheses) < highFi_budget:
                    normdiff = self._calculate_similarity(cand_fea, seed_data_fea)
                    if len(normdiff[(normdiff <= self.similarity_thres)]) >= 1:
                        selected_hypotheses = selected_hypotheses.append(high_fidelity_candidates.loc[idx])

            # query low fidelity candidate for remaining budget
            remained_highFi_cands_fea = highFi_cands_fea.drop(selected_hypotheses.index)
            lowFi_candidates_copy = low_fidelity_candidates.copy()
            for idx, cand_fea in remained_highFi_cands_fea.iterrows():
                if (len(selected_hypotheses) < self.total_budget) & (len(lowFi_candidates_copy) != 0):
                    lowFi_cands_fea = get_features_from_df(lowFi_candidates_copy, self.features)
                    lowFi_candidates_copy['normdiff'] = self._calculate_similarity(cand_fea, lowFi_cands_fea)
                    lowFi_candidates_copy = lowFi_candidates_copy.sort_values('normdiff')
                    selected_hypotheses = selected_hypotheses.append(lowFi_candidates_copy.head(self.lowFi_per_highFi))
                    lowFi_candidates_copy = lowFi_candidates_copy.drop(
                                            lowFi_candidates_copy.head(self.lowFi_per_highFi).index)
        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        """
        Gets hypotheses using agent.

        Args:
            candidate_data (pd.DataFrame): dataframe of candidates
            seed_data (pd.DataFrame): dataframe of known data

        Returns:
            (pd.DataFrame): dataframe of selected candidates

        """
        features_columns = self.features + list(self.fidelities)
        X_train, y_train, X_test, y_test = process_data(
            candidate_data, seed_data, features_columns, self.target_prop, preprocessor=self.preprocessor)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Make a copy of the candidate data so the original one
        # does not get modified during hypotheses generation
        candidate_data_copy = candidate_data.copy()
        candidate_data_copy['distance_to_ideal'] = np.abs(self.target_prop_val - y_pred)

        if self.ranking_method == 'minimize':
            candidate_data_copy = candidate_data_copy.sort_values(by=['distance_to_ideal'])
        elif self.ranking_method == 'ascending':
            candidate_data_copy = candidate_data_copy.sort_values(by=y_pred, ascending=True)
        elif self.ranking_method == 'descending':
            candidate_data_copy = candidate_data_copy.sort_values(by=y_pred, ascending=False)

        hypotheses = self._query_hypotheses(candidate_data_copy, seed_data)
        return hypotheses


class GPMultiAgent(HypothesisAgent):
    """
    A Gaussian process lower confidence bound derived multi-fidelity agent.
    This agent operates under a total acquisition budget. It acquires
    candidates factoring in GPR predicted uncertainties in the LCB setting
    and hallucination of information gain from DFT acquisitions analogous
    to work of Desautels et al. in batch mode LCB.  The agent aims for
    prioritizing exploitation primarily with high-fidelity experimental measurements,
    offloading exploratory (higher risk) acquisitions first to lower-fidelity computations.
    """
    def __init__(self, candidate_data=None, seed_data=None, chemsys_col='reduced_formula', features=None,
                 fidelities=('theory_data', 'expt_data'), target_prop=None, target_prop_val=1.8, total_budget=None,
                 preprocessor=StandardScaler(), gp_max_iter=200, alpha=1.0, rank_thres=10, unc_percentile=5):
        """
        Args:
            candidate_data (df)      Candidate search space for the campaign.
            seed_data (df)           Seed (training) data for the campaign.
            chemsys_col (str)        The name of the composition column.
            features (tuple of str)  Column name of the features used in machine learning.
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged
                                     from low to high fidelity. The value of fidelity features should
                                     be one-hot encoded.
            target_prop (str)        The property machine learning model is predicting, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            total_budget (int)       The budget for the hypotheses queried.
            gp_max_iter (int)        Number of maximum iterations for GP optimization.
            preprocessor             A preprocessor that preprocess the features. It can be None, a single
                                     processor, or a pipeline. The default is StandardScaler().
            alpha (float)            The mixing parameter for uncertainties. It controls the
                                     trade-off between exploration and exploitation. Defaults to 1.0.
            rank_thres (int)         A threshold help to decide if lower fidelity data is worth acquiring.
            unc_percentile (int)     A number between 0 and 100, and used to calculate an uncertainty threshold
                                     at that percentile value. The threshold is used to decide if the
                                     agent is quering theory or experimental hypotheses.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.chemsys_col = chemsys_col
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.total_budget = total_budget
        self.preprocessor = preprocessor
        self.gp_max_iter = gp_max_iter
        self.alpha = alpha
        self.rank_thres = rank_thres
        self.unc_percentile = unc_percentile
        super(GPMultiAgent).__init__()

    def _halluciate_lower_fidelity(self, seed_data, candidate_data, low_fidelity_data):
        # make copy of seed and candidate data, so we don't mess with the original one
        new_candidate_data = candidate_data.copy()
        low_fidelity = low_fidelity_data.copy()

        low_fidelity[self.target_prop] = low_fidelity_data['y_pred']
        low_fidelity = low_fidelity.drop(columns=['y_pred'])
        new_seed_data = pd.concat([seed_data, low_fidelity])
        new_candidate_data = new_candidate_data.drop(low_fidelity.index)
        pred_candidate_data = self._train_and_predict(new_candidate_data, new_seed_data)
        return pred_candidate_data

    def _get_rank_after_hallucination(self, seed_data, candidate_data, orig_idx, low_fidelity):
        halluciated_candidates = self._halluciate_lower_fidelity(seed_data, candidate_data, low_fidelity)
        halluciated_candidates = halluciated_candidates.loc[halluciated_candidates[self.fidelities[1]] == 1]
        halluciated_candidates = halluciated_candidates.sort_values('pred_lcb')
        rank_after_hallucination = halluciated_candidates.index.get_loc(orig_idx)
        return rank_after_hallucination

    def _train_and_predict(self, candidate_data, seed_data):
        features_columns = self.features + list(self.fidelities)
        X_train, y_train, X_test, y_test = process_data(
            candidate_data, seed_data, features_columns,
            self.target_prop, y_reshape=True, preprocessor=self.preprocessor
        )
        gp = GPy.models.GPRegression(X_train, y_train)
        gp.optimize('bfgs', max_iters=self.gp_max_iter)
        y_pred, var = gp.predict(X_test)

        # Make a copy of the candidate data so the original one
        # does not get modified during hypotheses generation
        candidate_data_copy = candidate_data.copy()
        dist_to_ideal = np.abs(self.target_prop_val - y_pred)
        pred_lcb = dist_to_ideal - self.alpha * var**0.5
        candidate_data_copy['pred_lcb'] = pred_lcb
        candidate_data_copy['pred_unc'] = var**0.5
        candidate_data_copy['y_pred'] = y_pred
        return candidate_data_copy

    def _query_hypotheses(self, candidate_data, seed_data):
        high_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[1]] == 1]
        high_fidelity_candidates = high_fidelity_candidates.sort_values('pred_lcb')

        # edge case: top the campaign if there are no high fidelity candidates
        if len(high_fidelity_candidates) == 0:
            return None

        else:
            selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
            unc_thres = np.percentile(np.array(high_fidelity_candidates.pred_unc), self.unc_percentile)

            # query hypothesis
            for idx, candidate in high_fidelity_candidates.iterrows():
                if len(selected_hypotheses) < self.total_budget:
                    chemsys = candidate[self.chemsys_col]
                    low_fidelity = candidate_data.loc[(candidate_data[self.chemsys_col] == chemsys) &
                                                      (candidate_data[self.fidelities[0]] == 1)]

                    # exploit if uncertainty is low or the low fidelity data is not available
                    if (candidate.pred_unc <= unc_thres) or (len(low_fidelity) == 0):
                        selected_hypotheses = selected_hypotheses.append(candidate)
                    # explore
                    else:
                        orig_rank = high_fidelity_candidates.index.get_loc(idx)
                        new_rank = self._get_rank_after_hallucination(seed_data, candidate_data, idx, low_fidelity)

                        delta_rank = new_rank - orig_rank
                        if delta_rank <= self.rank_thres:
                            selected_hypotheses = selected_hypotheses.append(candidate)
                        else:
                            selected_hypotheses = selected_hypotheses.append(low_fidelity)
        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        """
        Selects candidate data for experiment using agent methods

        Args:
            candidate_data (pd.DataFrame): candidate data
            seed_data (pd.DataFrame): known data on which to base the selection
                of candidates

        Returns:
            (pd.DataFrame): selected candidates e.g. for experiment

        """
        candidate_data = self._train_and_predict(candidate_data, seed_data)
        hypotheses = self._query_hypotheses(candidate_data, seed_data)
        return hypotheses
