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
from sklearn import preprocessing

from camd.agent.base import HypothesisAgent, RandomAgent
from camd.analysis import AnalyzerBase
from camd.experiment.base import ATFSampler
from camd.campaigns.base import Campaign
from camd import CAMD_CACHE


# Agent Design
# ----------------------------------------------------------------------
class MultiAgent(HypothesisAgent):
    def __init__(self, target_property='bandgap', ideal_property_value=1.8,
                 candidate_data=None, seed_data=None, n_query=None, exp_query_frac=0.3,
                 model=None, preprocessor=None,
                 cost_considered=False, uncertainty=True):
        self.target_property = target_property
        self.ideal_property_value = ideal_property_value
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query
        self.exp_query_frac = exp_query_frac
        self.model = model
        self.preprocessor = preprocessor
        self.cost_considered = cost_considered
        self.uncertainty = uncertainty
        super(MultiAgent).__init__()
        """
        Args:
            target_property          The candidate property agent is trying to predict on,
                                     given feature space. i.e. bandgap.
            target_property_value    The ideal value of the target property.
            candidate_data           A pd.DataFrame of candidate search space learning.
            seed_data                A pd.DataFrame of training data for learning.
            N_query (int)            Number of hypotheses to generate.
            exp_query_frac           The fraction of the quries that are experimental data
            model                    The machine learning model.
            preprocessor             The preprocessor that preprocess the feature space.
                                     It can be a single step or a pipeline.
            cost_considered          If True, hypotheses are generated based on both prediction value
                                     and the cost of generating such hypotheses.
            uncertainty              If we want to use ML model such as GP, this arg should be True.
        """

    def get_features_from_df(self, df, add_fea=[]):
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

    def get_hypotheses(self, candidate_data, seed_data):
        # Fit the ML model
        X_train = self.get_features_from_df(seed_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        y_train = np.array(seed_data[[self.target_property]])
        X_test = self.get_features_from_df(candidate_data, add_fea=['theory_data', 'expt_data']).values.tolist()
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)

        # make predictions
        if self.uncertainty:
            gp = GPy.models.GPRegression(X_train, y_train)
            gp.optimize('bfgs', max_iters=200)
            y_pred, unc = gp.predict(X_test)
            candidate_data['prediction'] = y_pred
            candidate_data['pred_unc'] = unc

            # Generate hypotheses
            candidate_data['dist_to_ideal'] = np.abs(self.ideal_property_value - candidate_data['prediction'])
            candidate_data = candidate_data.sort_values(by=['dist_to_ideal', 'pred_unc'])

        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            candidate_data['prediction'] = y_pred

            # Generate hypotheses
            candidate_data['dist_to_ideal'] = np.abs(self.ideal_property_value - candidate_data['prediction'])
            candidate_data = candidate_data.sort_values(by=['dist_to_ideal'])

        selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)

        # First: select expt hypotheses that have enough structurally similar seed data
        # that support their predicted label
        seed_data_fea = self.get_features_from_df(seed_data)
        exp_candidates = candidate_data.loc[candidate_data.expt_data==1]
        exp_cands_fea = self.get_features_from_df(exp_candidates)

        for idx, cand_fea in exp_cands_fea.iterrows():
            if len(selected_hypotheses) < self.n_query * self.exp_query_frac:
                # find l2 norm of features between this candidate and all seed data
                # and find the ones that are closest "structurally"
                expanded_cand_fea = pd.DataFrame([cand_fea]*len(seed_data_fea))
                normdiff = np.linalg.norm(expanded_cand_fea.values - seed_data_fea.values,  axis=1)

                # TODO: make threshold parameter
                # TODO: decide how many similar structure should be there
                threshold = 200
                mask = normdiff <= threshold
                if len(normdiff[mask]) > 2:
                    selected_hypotheses = selected_hypotheses.append(exp_candidates.loc[idx])


        # Genetate some theory hypotheses, which has good predictions and structurally similar
        # to the top experimental predicted candidates
        remained_exp_cands_fea = exp_cands_fea.drop(selected_hypotheses.index)

        theor_candidates = candidate_data.loc[candidate_data.theory_data==1]
        theor_cands_fea = self.get_features_from_df(theor_candidates)

        for idx, cand_fea in remained_exp_cands_fea.iterrows():
            if len(selected_hypotheses) < self.n_query:
                expanded_cand_fea = pd.DataFrame([cand_fea]*len(theor_cands_fea))
                normdiff = np.linalg.norm(expanded_cand_fea.values - theor_cands_fea.values,  axis=1)
                import nose; nose.tools.set_trace()
                theor_candidates['normdiff'] = normdiff
                theor_candidates = theor_candidates.sort_values('normdiff')
                selected_hypotheses = selected_hypotheses.append(theor_candidates.head(4))

        return selected_hypotheses

# Experiment
# -----------------------------------------------------------------------
# ATFSampler from CAMD is used


# Analyzer
# -----------------------------------------------------------------------
class MultiAnalyzer(AnalyzerBase):
    def __init__(self, target_property, property_range):
        self.target_property = target_property
        self.property_range = property_range

    def _filter_df_by_property_range(self, df):
        """
        Helper function to filter df by property range

        Args:
            df (DataFrame): dataframe to be filtered
        """
        return df.loc[(df[self.target_property] >= self.property_range[0]) &
                      (df[self.target_property] <= self.property_range[1])]

    def analyze(self, new_experimental_results, seed_data):  #, agent=None):
        positive_hits = self._filter_df_by_property_range(new_experimental_results)

        new_exp_hypotheses = new_experimental_results.loc[new_experimental_results['expt_data'] == 1]
        new_discoveries = self._filter_df_by_property_range(new_exp_hypotheses)
        # Uncertainty?

        # total discovery up to (& including) the current iteration
        new_seed = seed_data.append(new_experimental_results)
        total_exp_hypotheses = new_seed.loc[new_seed['expt_data'] == 1]
        total_exp_discovery = self._filter_df_by_property_range(total_exp_hypotheses)

        summary = pd.DataFrame(
            {   "iteration tpr": [len(positive_hits)/ len(new_experimental_results)],
                "iteration experiment tpr": [len(new_discoveries)/ len(new_exp_hypotheses)],
                "new_exp_discovery": [len(new_discoveries)],
                "total_exp_discovery": [len(total_exp_discovery)]
                # "discoveries_per_cost": [len(new_discovery) / cost]
            }
        )
        return summary, new_seed

if __name__ == "__main__":
    # Put everything together
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Decide on campaign
    campaign_type = "multi-svr"
    campaign_iterations = 10
    N_query = 20

    # Load dataset
    # ---------------------------------------------------------------------
    featurized_data = pd.read_csv(os.path.join(CAMD_CACHE, "brgoch_featurized_data.csv"), index_col=0)
    seed_data, candidate_data = train_test_split(featurized_data, test_size=0.8, random_state=42)
    # Drop all the candidate compositions if they are already verified in seed data.
    seed_data_chemsys = list(seed_data.reduced_formula)
    seed_data  = seed_data.append(candidate_data.loc[candidate_data.reduced_formula.isin(seed_data_chemsys)])
    candidate_data = candidate_data[~candidate_data.reduced_formula.isin(seed_data_chemsys)]

    print(len(seed_data)+len(candidate_data)==len(featurized_data))

    # set up params for campaign
    # ---------------------------------------------------------------------
    if campaign_type == 'random':
        agent = RandomAgent(candidate_data=candidate_data, n_query=N_query)

    elif campaign_type == 'multi-svr':
        agent = MultiAgent(target_property='bandgap', ideal_property_value=1.8,
                                 candidate_data=candidate_data, seed_data=seed_data, n_query=N_query,
                                 model=SVR(C=10), preprocessor=preprocessing.StandardScaler())

    elif campaign_type == 'multi-gp':
        agent = MultiAgent(target_property='bandgap', ideal_property_value=1.8,
                                 candidate_data=candidate_data, seed_data=seed_data, n_query=N_query,
                                 preprocessor=preprocessing.StandardScaler(), uncertainty=True)

    experiment = ATFSampler(dataframe=featurized_data)
    analyzer = MultiAnalyzer(target_property='bandgap', property_range=[1.6, 2.0])

    # Run the campaign
    os.system('rm -rf {}'.format(campaign_type))
    os.system('mkdir -p {}'.format(campaign_type))
    with cd(campaign_type):
        multi_campaign = Campaign(candidate_data=candidate_data, seed_data=seed_data,
                                  agent=agent, experiment=experiment,
                                  analyzer=analyzer)
        multi_campaign.auto_loop(n_iterations=campaign_iterations, initialize=True)



