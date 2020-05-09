import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from camd.agent.base import HypothesisAgent


class GenericGPUCB(HypothesisAgent):
    def __init__(self, candidate_data=None, seed_data=None, n_query=None, alpha=1.0, kernel=None):
        """
        Generic GP-UCB agent that tries to maximize a target.
        candidate_data: dataframe of candidate features.
        seed_data: dataframe of seed data. It has to have a "target" column.
        n_query: allowed acquisition budget in each iteration
        alpha: mixing parameter for GP-UCB
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.cv_score = np.inf
        self.alpha = alpha
        self.kernel = kernel if kernel else \
            ConstantKernel(1.0)*RBF(1.0)
        super(GenericGPUCB).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.candidate_data = candidate_data.drop(columns=['target'], axis=1)
        self.seed_data = seed_data
        X_seed = seed_data.drop(columns=['target'], axis=1)
        y_seed = seed_data['target']
        steps = [('scaler', StandardScaler()),
                 ('GP', GaussianProcessRegressor(kernel=self.kernel, normalize_y=True,
                                                 n_restarts_optimizer=25))]
        self.pipeline = Pipeline(steps)
        self.cv_score = np.mean(-1 * cross_val_score(self.pipeline, X_seed, y_seed,
                                                     cv=KFold(3, shuffle=True)))
        self.pipeline.fit(X_seed, y_seed)
        t_pred, unc = self.pipeline.predict(self.candidate_data, return_std=True)
        t_pred += unc * self.alpha
        selected = np.argsort(-1.0 * t_pred)[:self.n_query]
        return candidate_data.loc[selected]