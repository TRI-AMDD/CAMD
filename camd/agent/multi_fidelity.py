# Copyright Toyota Research Institute 2019
import GPy
import numpy as np
import pandas as pd

from monty.os import cd
from sklearn.preprocessing import StandardScaler
from camd.agent.base import HypothesisAgent


def preprocess(dataframes, composition_columns, featurizer, fill_na=-1,label_columns=None):
    """
    A hlper function that preprocess multi-fidelity datasets. It will take multiple 
    datasets (in dataframe form), and featuriized the compositional features, and 
    one-hot encode any other additional features. Lastly, it will fill in null features. 
    
    Args:
        dataframes (list)            A list of pd.DataFrames where each dataframe is a dataset.
                                        Note, fidelity level should be included. 
        composition_columns (list)   A list of compositional feature column names in corresponding
                                        dataset.
        featurizer                   Featurizer used to featurize the compositional feature. 
        fill_na (float)              The number used to fill null values in the dataframe. 
        label_columns (list)         A list of additional features combined from each dataframe. 
    
    Returns:
        all_data                     A processed pd.DataFrame.
    """
    all_features = []
    for df, comp_col in zip(dataframes, composition_columns):
        if label_columns is None:
            extra_label = list(set(df.columns) - {comp_col})
        df['composition'] = df[comp_col].apply(Composition)  
        featurized = featurizer.featurize_dataframe(df, 'composition')
        one_hot_df = pd.get_dummies(df[extra_label])
        all_features.append(pd.concat(
                                      [featurized[featurizer.feature_labels()], 
                                      one_hot_df], axis=1))
    all_data = pd.concat(all_features, axis=0)
    all_data = all_data.fillna(fill_na)
    return all_data  



class GenericMultiAgent(HypothesisAgent):
    """
    A multi-fidelity agent that takes in sklearn supervised regressor
    (GPR excluded) and generate hypotheses.
    """
    def __init__(self, candidate_data=None, seed_data=None, features=None, fidelities=('theory_data', 'expt_data'),
                 target_prop=None, target_prop_val=None, preprocessor=StandardScaler(), model=None, minimize=True,
                 query_criteria='number', total_budget=None, expt_query_frac=None, l2norm_thres=300., theor_per_expt=1):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.preprocessor = preprocessor
        self.model = model
        self.minimize = minimize
        self.query_criteria = query_criteria
        self.total_budget = total_budget
        self.expt_query_frac = expt_query_frac
        self.l2norm_thres = l2norm_thres
        self.theor_per_expt = theor_per_expt
        super(GenericMultiAgent).__init__()
        """
        Args:
            candidate_data (df)      Candidate search space for active learning.
            seed_data (df)           Seed (training) data for active learning.
            features (tuple of str)  Name of the features used in machine learning. If None, default features
                                     (magpie features) are used. 
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged 
                                     from low to high fidelity. The value of fidelity features should be one-hot encoded.    
            target_prop (str)        The property machine learning model is predicting, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            preprocessor             A sklearn preprocessor that preprocess the features. It can be None, a single  
                                     processor, or a pipeline. The default is StandardScaler().
            model                    The sklearn supervised machine learning regressor (GPR excluded).
            minimize (bool)          If True, rank candidates with ML prediction as close to the target property value
                                     as possible. If False, rank candidates with ML prediction as far away from the target 
                                     as possible. Minimize=False will be helpful when there is no target value for a 
                                     prediction problem, such as bulk modulus. 
            query_criteria (str)     Can be either 'cost_ratio' or 'number' to indicate which types of method
                                     will be used to query hypotheses. .                        
            total_budget (int)       The budget for the hypotheses query. If query_criteria is 'cost_ratio', this should 
                                     be an amount indicates cost, if query_criteria is 'number', this should be a 
                                     a total number of queries.
            expt_query_frac (float)  The fraction of the total budget that are used for experimental
                                     hypotheses queries. The value should be between 0 and 1.
            l2norm_thres(float)      The threshold value for l2 norm similarity between a candidate composition and
                                     compositions in the seed data.  
            theor_per_expt (int)     Theory hypotheses budget per experimental candidates. In other
                                     words, the number of theory candidate selected to support each
                                     experimental candidates that predicted to be good, but the agent
                                     does not want to generate that experimental hypotheses yet. 
        """

    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.
    
        Args:
            df              A pd.DataFrame where the features are extracted.
            add_fea(list)   Name of the additional features (str) used in ML.
    
        Returns:
            feature_df      A pd.DataFrame that only contains the features used in ML.
                                by default, this will be compositional features.
        """
        if self.features is None:
            features = [column for column in df if column.startswith("MagpieData")]
        else:
            features = list(self.features)
            
        all_features = features + add_fea
        feature_df = df[all_features]
        return feature_df

    def _process_data(self, candidate_data, seed_data):
        """
        Helper function that process data for ML model and returns
        np.ndarray of training features, training labels,
        test features and test labels.
        """
        X_train = self._get_features_from_df(seed_data, add_fea=list(self.fidelities)).values
        y_train = np.array(seed_data[self.target_prop])
        X_test = self._get_features_from_df(candidate_data, add_fea=list(self.fidelities)).values
        y_test = np.array(candidate_data[self.target_prop])
      
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test

    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between
        a composition and the seed data compositions. The similarity
        is reprsented by l2_norm.

        Args:
            comp(pd.core.series):    A specific composition represented by Magpie.
            seed_comps (df):         Compostions in seed represented by Magpie.
        """
        # match dimension to element wise operations
        comp = pd.DataFrame([comp]*len(seed_comps))
        l2_norm = np.linalg.norm(comp.values - seed_comps.values,  axis=1)
        return l2_norm

    def _query_hypotheses(self, candidate_data, seed_data):
        exp_candidates = candidate_data.loc[candidate_data.expt_data == 1]
        theor_candidates = candidate_data.loc[candidate_data.theory_data == 1]
        seed_data_fea = self._get_features_from_df(seed_data)
        exp_cands_fea = self._get_features_from_df(exp_candidates)

        # set up an empty df for queries
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
        
        # if there are no experimental data left in the candidate space, end the campaign
        if len(exp_candidates) == 0:
            return None
         
        # if there are only experimental data left in the candidate space 
        # use the entire iteration budget on experimental queries
        elif (len(exp_candidates) != 0) & (len(theor_candidates) == 0):
            for idx, exp in exp_candidates.iterrows(): 
                if self.query_criteria == 'number':
                    total_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    total_cost = np.sum(selected_hypotheses[self.query_criteria])

                if total_cost < self.total_budget:
                    selected_hypotheses = selected_hypotheses.append(exp)

        # query both experimental and theory hypotheses
        else:
            # first query experimental candidates that have support 
            # from the seed data (i.e. similar compostion). 
            exp_budget = self.total_budget * self.expt_query_frac
            l2norm_thres = self.l2norm_thres
            for idx, cand_fea in exp_cands_fea.iterrows(): 
                if self.query_criteria == 'number':
                    expt_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    expt_cost = np.sum(selected_hypotheses[self.query_criteria])
                
                if expt_cost < exp_budget:
                    normdiff_lst = self._calculate_similarity(cand_fea, seed_data_fea)
                    mask = (normdiff_lst <= l2norm_thres)
                    if len(normdiff_lst[mask]) >= 1:
                        selected_hypotheses = selected_hypotheses.append(exp_candidates.loc[idx])

            # for remaining of the budget, select DFT hypotheses that supports 
            # experimental data that predicted to be ideal. 
            remained_exp_cands_fea = exp_cands_fea.drop(selected_hypotheses.index)
            theor_candidates_copy = theor_candidates.copy()
            for idx, cand_fea in remained_exp_cands_fea.iterrows():
                if self.query_criteria == 'number':
                    total_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    total_cost = np.sum(selected_hypotheses[self.query_criteria])
                    
                if (total_cost < self.total_budget) & (len(theor_candidates_copy) !=0):
                    theor_cands_fea = self._get_features_from_df(theor_candidates_copy)
                    theor_candidates_copy['normdiff'] = self._calculate_similarity(cand_fea, theor_cands_fea) 
                    theor_candidates_copy = theor_candidates_copy.sort_values('normdiff')
                    selected_hypotheses = selected_hypotheses.append(theor_candidates_copy.head(self.theor_per_expt))
                    theor_candidates_copy = theor_candidates_copy.drop(theor_candidates_copy.head(self.theor_per_expt).index)
        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        candidate_data_copy = candidate_data.copy()
        X_train, y_train, X_test, y_test = self._process_data(candidate_data_copy, seed_data)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        candidate_data_copy['dist_to_ideal'] = np.abs(self.target_prop_val - y_pred)
        if self.minimize:
            candidate_data_copy = candidate_data_copy.sort_values(by=['dist_to_ideal'])
        else:
            candidate_data_copy = candidate_data_copy.sort_values(by=['dist_to_ideal'], ascending=False)
        hypotheses = self._query_hypotheses(candidate_data_copy, seed_data)
        return hypotheses


class GPMultiAgent(HypothesisAgent):
    """
    Similar to Generic_MultiAgent, but the ML model is Gaussian Process regressor
    from GPy). This agent accounts for uncertainty, and use the lower confidence bound approach 
    to generate hypotheses.
    """
    def __init__(self, candidate_data=None, seed_data=None, chemsys_col='reduced_formula', features=None, 
                 fidelities=('theory_data', 'expt_data'), target_prop=None, target_prop_val=1.8, 
                 preprocessor=StandardScaler(), alpha=0.5, rank_thres=5, unc_percentile=30, 
                 query_criteria='cost_ratio', total_budget=None):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.chemsys_col = chemsys_col
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.rank_thres = rank_thres
        self.unc_percentile = unc_percentile
        self.query_criteria = query_criteria
        self.total_budget = total_budget
        super(GPMultiAgent).__init__()

        """
        Args:
            candidate_data (df)      Candidate search space for active learning.
            seed_data (df)           Seed (training) data for active learning.
            chemsys_col (str)        The name of the composition column.
            features (tuple of str)  Name of the features used in machine learning. If None, default features
                                     (magpie features) are used. 
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged 
                                     from low to high fidelity. The value of fidelity features should be one-hot encoded.    
            target_prop (str)        The property machine learning model is predicting, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            preprocessor             A sklearn preprocessor that preprocess the features. It can be None, a single  
                                     processor, or a pipeline. The default is StandardScaler().
            alpha (float)            The mixing parameter for uncertainties. It controls the 
                                     trade-off between exploration and exploitation. Defaults to 1.0. 
            rank_thres (int)         A threshold help to decide if lower fidelity data is worth acquiring.                 
            unc_percentile (int)     A number between 0 and 100, and used to calculate an uncertainty threshold
                                     at that percentile value. The threshold is then used to decide if the
                                     agent is quering theory or experimental hypotheses. 
            query_criteria (str)     Can be either 'cost_ratio' or 'number' to indicate which types of method
                                     will be used to query hypotheses. .                        
            total_budget (int)       The budget for the hypotheses query. If query_criteria is 'cost_ratio', this should 
                                     be an amount indicate costs, if query_criteria is 'number', this should be a 
                                     a total number of queries.
            """
    def _get_features_from_df(self, df, add_fea=[]):
        """
        Helper function to get feature columns of a dataframe.
    
        Args:
            df              A pd.DataFrame where the features are extracted.
            add_fea(list)   Name of the additional features (str) used in ML.
    
        Returns:
            feature_df      A pd.DataFrame that only contains the features used in ML.
                                by default, this will be compositional features.
        """
        if self.features is None:
            features = [column for column in df if column.startswith("MagpieData")]
        else:
            features = list(self.features)
            
        all_features = features + add_fea
        feature_df = df[all_features]
        return feature_df

    def _process_data(self, candidate_data, seed_data):
        """
        Helper function that process data for ML model and returns
        np.ndarray of training features, training labels,
        test features and test labels. Note, Y.ndim == 2
        """
        X_train = self._get_features_from_df(seed_data, add_fea=list(self.fidelities)).values
        y_train = np.array(seed_data[[self.target_prop]])
        X_test = self._get_features_from_df(candidate_data, add_fea=list(self.fidelities)).values
        y_test = np.array(candidate_data[[self.target_prop]])
      
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        return X_train, y_train, X_test, y_test
        
    def _halluciate_lower_fidelity(self, seed_data, candidate_data, low_fidelity_data):
        # make copy of seed and candidate data, so we don't mess with the original one
        new_seed_data = seed_data.copy()
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
        halluciated_candidates = halluciated_candidates.loc[halluciated_candidates[self.fidelities[1]]== 1]
        halluciated_candidates = halluciated_candidates.sort_values('pred_lcb')
        rank_after_hallucination = halluciated_candidates.index.get_loc(orig_idx)
        return rank_after_hallucination

    def _train_and_predict(self, candidate_data, seed_data):
        candidate_data_copy = candidate_data.copy()
        X_train, y_train, X_test, y_test = self._process_data(candidate_data_copy, seed_data)
        gp = GPy.models.GPRegression(X_train, y_train)
        gp.optimize('bfgs', max_iters=200)
        y_pred, var = gp.predict(X_test)
        dist_to_ideal = np.abs(self.target_prop_val - y_pred)
        pred_lcb = dist_to_ideal - self.alpha * var**0.5 
        candidate_data_copy['pred_lcb'] = pred_lcb 
        candidate_data_copy['pred_unc'] = var**0.5 
        candidate_data_copy['y_pred'] = y_pred 
        return candidate_data_copy
        
    def _query_hypotheses(self, candidate_data, seed_data): 
        high_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[1]]== 1]
        high_fidelity_candidates = high_fidelity_candidates.sort_values('pred_lcb')
        
        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
        if len(high_fidelity_candidates) == 0:
            return None

        else:
            unc_thres = np.percentile(np.array(high_fidelity_candidates.pred_unc), self.unc_percentile)
            for idx, candidate in high_fidelity_candidates.iterrows():
                if self.query_criteria == 'number':
                    total_cost = len(selected_hypotheses)
                elif self.query_criteria == 'cost_ratio':
                    total_cost = np.sum(selected_hypotheses[self.query_criteria])

                # query more hypotheses if total budget is not fulfilled
                if total_cost < self.total_budget:
                    chemsys = candidate[self.chemsys_col]
                    low_fidelity = candidate_data.loc[(candidate_data[self.chemsys_col] == chemsys)&
                                                      (candidate_data[self.fidelities[0]] == 1)]
                    # exploit
                    if (candidate.pred_unc <= unc_thres) or (len(low_fidelity) == 0):
                        selected_hypotheses = selected_hypotheses.append(candidate)
                    # explore
                    else:
                        orig_rank = high_fidelity_candidates.index.get_loc(idx)
                        new_rank = self._get_rank_after_hallucination(seed_data, candidate_data, idx, low_fidelity)
                        delta_rank = new_rank - orig_rank
                        # If adding predicted DFT value to the seed data confirms the candidate
                        # is good, we go ahead and query the candidate. If not, we query the lower
                        # fidelity first to the seed space. 
                        if delta_rank <= self.rank_thres:
                            selected_hypotheses = selected_hypotheses.append(candidate)
                        else:
                            selected_hypotheses = selected_hypotheses.append(low_fidelity)
        return selected_hypotheses     

    def get_hypotheses(self, candidate_data, seed_data):
        candidate_data = self._train_and_predict(candidate_data, seed_data)
        hypotheses = self._query_hypotheses(candidate_data, seed_data)
        return hypotheses
