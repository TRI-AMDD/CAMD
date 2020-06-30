"""
Prototype multi-fidelity campaign
"""

# Import packages
# --------------------------------------------------------------------
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

# Set pandas view options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from monty.os import cd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from camd.agent.base import HypothesisAgent, RandomAgent
from camd.analysis import AnalyzerBase
from camd.experiment.base import ATFSampler
from camd.campaigns.base import Campaign
from camd import CAMD_CACHE


class GenericMultiAgent(HypothesisAgent):
    def __init__(self, target_prop='bandgap', ideal_prop_val=1.8,
                 candidate_data=None, seed_data=None, preprocessor=StandardScaler(),
                 model=None, n_query=None, exp_query_frac=0.2, cost_considered=False):
        self.target_prop = target_prop
        self.ideal_prop_val = ideal_prop_val
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.preprocessor = preprocessor
        self.model = model
        self.n_query = n_query
        self.exp_query_frac = exp_query_frac
        self.cost_considered = cost_considered
        super(GenericMultiAgent).__init__()
        """
        Args:
            target_prop              The property agent is trying to predict on,
                                     given feature space. i.e. bandgap.
            target_prop_val          The ideal value of the target property.
            candidate_data           A pd.DataFrame of candidate search space learning.
            seed_data                A pd.DataFrame of training data for learning.
            preprocessor             The preprocessor that preprocess the feature space.
                                     It can be a single step or a pipeline.  
            model                    The machine learning model.                          
            N_query (int)            Number of hypotheses to generate.
            exp_query_frac           The fraction of the quries that are experimental data
            cost_considered          If True, hypotheses are generated based on both prediction value
                                     and the cost of generating such hypotheses.
        """

    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.

        Args
            df           df where the features are extracted.
            add_fea      additional features used in ML training

        Returns
            feature_df   A dataframe that only contains the features used in ML.
                         by default, this will be compositional features.
        """
        magpie_columns = [column for column in df if column.startswith("MagpieData")]
        all_features = magpie_columns + add_fea
        feature_df = df[all_features]
        return feature_df 

    def _process_data(self, candidate_data, seed_data):
        """
        process data for ML training. 
        """
        X_train = self._get_features_from_df(seed_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_train = np.array(seed_data[[self.target_prop]])
        X_test = self._get_features_from_df(candidate_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_test = np.array(candidate_data[[self.target_prop]])
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test
    
    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between a composition 
        and the seed data compositions.
        The similarity is reprsented by l2_norm. 
        
        Args: 
            comp(pd.core.series):    A specific composition represented by Magpie.              
            seed_comps (df):         Compostions in seed represented by Magpie. 
        """
        # match dimension to element wise operations
        comp = pd.DataFrame([comp]*len(seed_comps))
        l2_norm = np.linalg.norm(comp.values - seed_comps.values,  axis=1)
        return l2_norm

    def _select_hypotheses(self, candidate_data, seed_data):
        seed_data_fea = self._get_features_from_df(seed_data)
        
        # set up an empty df
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
        
        # select experimental hypotheses
        num_exp_query = self.n_query * self.exp_query_frac
        exp_candidates = candidate_data.loc[candidate_data.expt_data == 1]
        exp_cands_fea = self._get_features_from_df(exp_candidates)
        
        threshold = 200 # TODO, make threshold params 
        for idx, cand_fea in exp_cands_fea.iterrows(): #TODO: fix the edge case, len(exp_cands_fea)=0
            if len(selected_hypotheses) < num_exp_query:
                normdiff = self._calculate_similarity(cand_fea, seed_data_fea)
                if len(normdiff <= threshold) > 1:
                    selected_hypotheses = selected_hypotheses.append(exp_candidates.loc[idx])

        # select DFT hypotheses
        remained_exp_cands_fea = exp_cands_fea.drop(selected_hypotheses.index)
        theor_candidates = candidate_data.loc[candidate_data.theory_data == 1]
        theor_cands_fea = self._get_features_from_df(theor_candidates)
        
        for idx, cand_fea in remained_exp_cands_fea.iterrows():
            if len(selected_hypotheses) < self.n_query:
                theor_candidates['normdiff'] = self._calculate_similarity(cand_fea, theor_cands_fea)
                theor_candidates = theor_candidates.sort_values('normdiff')
                selected_hypotheses = selected_hypotheses.append(theor_candidates.head(4))
        return selected_hypotheses
                                    
    def _select_hypotheses_with_cost(self, candidate_data, seed_data):
        pass
  
    def get_hypotheses(self, candidate_data, seed_data):
        X_train, y_train, X_test, y_test = self._process_data(candidate_data, seed_data)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        candidate_data['prediction'] = y_pred
        candidate_data['dist_to_ideal'] = np.abs(self.ideal_prop_val - candidate_data['prediction'])    
        candidate_data = candidate_data.sort_values(by=['dist_to_ideal'])
        if self.cost_considered:
            hypotheses = self._select_hypotheses_with_cost()
        else:
            hypotheses = self._select_hypotheses(candidate_data, seed_data)
        return hypotheses
    
        
class GPMultiAgent(HypothesisAgent):
    """
    Similar to Generic_MultiAgent, but the ML model is Gaussian Process. 
    This agent accounts for uncertainty when selecting candidates. 
    """
    def __init__(self, target_prop='bandgap', ideal_prop_val=1.8, 
                 candidate_data=None, seed_data=None, preprocessor=None,  
                 n_query=None, exp_query_frac=0.2, cost_considered=False):
        self.target_prop = target_prop
        self.ideal_prop_val = ideal_prop_val
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.preprocessor = preprocessor
        self.n_query = n_query
        self.exp_query_frac = exp_query_frac
        self.cost_considered = cost_considered
        super(GPMultiAgent).__init__()
        
        """
        Args:
            target_prop              The candidate property agent is trying to predict on,
                                     given feature space. i.e. bandgap.
            target_prop_val          The ideal value of the target property.
            candidate_data           A pd.DataFrame of candidate search space learning.
            seed_data                A pd.DataFrame of training data for learning.
            preprocessor             The preprocessor that preprocess the feature space.
                                     It can be a single step or a pipeline.    
            N_query (int)            Number of hypotheses to generate.
            exp_query_frac           The fraction of the quries that are experimental data   
            cost_considered          If True, hypotheses are generated based on both prediction value
                                     and the cost of generating such hypotheses.
        """
    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.

        Args
            df           df where the features are extracted.
            add_fea      additional features used in ML training

        Returns
            feature_df   A dataframe that only contains the features used in ML.
                         by default, this will be compositional features.
        """
        magpie_columns = [column for column in df if column.startswith("MagpieData")]
        all_features = magpie_columns + add_fea
        feature_df = df[all_features]
        return feature_df 

    def _process_data(self,  candidate_data, seed_data):
        """
        process data for ML training. 
        """
        X_train = self._get_features_from_df(seed_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_train = np.array(seed_data[[self.target_prop]])
        X_test = self._get_features_from_df(candidate_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_test = np.array(candidate_data[[self.target_prop]])
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test

    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between a composition 
        and the seed data compositions.
        The similarity is reprsented by l2_norm. 
        
        Args: 
            comp(pd.core.series):    a specific composition represented by Magpie.              
            seed_comps (df):         Compostions in seed represented by Magpie. 
        """
        # match dimension to element wise operations
        comp = pd.DataFrame([comp]*len(seed_comps))
        l2_norm = np.linalg.norm(comp.values - seed_comps.values,  axis=1)
        return l2_norm
  
    
    def _select_hypotheses(self, candidate_data, seed_data):
        seed_data_fea = self._get_features_from_df(seed_data)
        
        # set up an empty df
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
        
        # select experimental hypotheses - source from candidates with prediction
        # close to the ideal value and have low uncertainty.
        num_exp_query = self.n_query * self.exp_query_frac
        unc_thres = np.percentile(np.array(candidate_data.pred_unc), 30)  #TODO
        exp_candidates = candidate_data.loc[(candidate_data.expt_data == 1)&
                                            (candidate_data.pred_unc <= unc_thres)]
        exp_cands_fea = self._get_features_from_df(exp_candidates)
        
        threshold = 200 # TODO, make threshold params 
        for idx, cand_fea in exp_cands_fea.iterrows(): #TODO: fix the edge case, len(exp_cands_fea)=0
            if len(selected_hypotheses) < num_exp_query:
                normdiff = self._calculate_similarity(cand_fea, seed_data_fea)
                if len(normdiff <= threshold) > 1:
                    selected_hypotheses = selected_hypotheses.append(exp_candidates.loc[idx])

        # For experimental prediction with high uncertainty, select DFT hypotheses
        remained_exp_candidates = candidate_data.loc[(candidate_data.expt_data == 1)&
                                                     (candidate_data.pred_unc > unc_thres)]
        theor_candidates = candidate_data.loc[candidate_data.theory_data == 1]
        
        remained_exp_cands_fea = self._get_features_from_df(remained_exp_candidates)
        theor_cands_fea = self._get_features_from_df(theor_candidates)
        
        for idx, cand_fea in remained_exp_cands_fea.iterrows():
            if len(selected_hypotheses) < self.n_query:
                theor_candidates['normdiff'] = self._calculate_similarity(cand_fea, theor_cands_fea)
                theor_candidates = theor_candidates.sort_values('normdiff')
                selected_hypotheses = selected_hypotheses.append(theor_candidates.head(4))
        return selected_hypotheses
                                    
    def _select_hypotheses_with_cost(self, candidate_data, seed_data):
        pass
  
    def get_hypotheses(self, candidate_data, seed_data):
        X_train, y_train, X_test, y_test = self._process_data(candidate_data, seed_data)
        gp = GPy.models.GPRegression(X_train, y_train)
        gp.optimize('bfgs', max_iters=200)
        y_pred, var = gp.predict(X_test)
        candidate_data['prediction'] = y_pred
        candidate_data['dist_to_ideal'] = np.abs(self.ideal_prop_val - y_pred)  
        candidate_data['pred_unc'] = var**0.5
  
        candidate_data = candidate_data.sort_values(by=['dist_to_ideal'])
        if self.cost_considered:
            hypotheses = self._select_hypotheses_with_cost()
        else:
            hypotheses = self._select_hypotheses(candidate_data, seed_data)
        return hypotheses


# -----------------------------------------------------------------------
class MultiAnalyzer(AnalyzerBase):
    def __init__(self, target_prop, prop_range):
        self.target_prop = target_prop
        self.prop_range = prop_range

    def _filter_df_by_prop_range(self, df):
        """
        Helper function to filter df by property range

        Args:
            df (DataFrame): dataframe to be filtered
        """
        return df.loc[(df[self.target_prop] >= self.prop_range[0]) &
                      (df[self.target_prop] <= self.prop_range[1])]

    def analyze(self, new_experimental_results, seed_data):  #, agent=None):
        positive_hits = self._filter_df_by_prop_range(new_experimental_results)

        new_exp_hypotheses = new_experimental_results.loc[new_experimental_results['expt_data'] == 1]
        new_discoveries = self._filter_df_by_prop_range(new_exp_hypotheses)
        # Uncertainty?

        # total discovery up to (& including) the current iteration
        new_seed = seed_data.append(new_experimental_results)
        total_exp_hypotheses = new_seed.loc[new_seed['expt_data'] == 1]
        total_exp_discovery = self._filter_df_by_prop_range(total_exp_hypotheses)

        summary = pd.DataFrame(
            {   "iteration tpr": [len(positive_hits)/ len(new_experimental_results)],
#                 "iteration experiment tpr": [len(new_discoveries)/ len(new_exp_hypotheses)],
                "new_exp_discovery": [len(new_discoveries)],
                "total_exp_discovery": [len(total_exp_discovery)]
                # "discoveries_per_cost": [len(new_discovery) / cost]
            }
        )
        return summary, new_seed