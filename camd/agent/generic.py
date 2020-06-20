import numpy as np
import gpflow

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from camd.agent.base import HypothesisAgent


class GenericGPUCB(HypothesisAgent):
    """
    Generic Gaussian Process (GP) and upper confidence bound (UCB)
    based agent that tries to maximize a target that can
    be used in a batch-mode Bayesian Optimization setting.
    """

    def __init__(
        self, candidate_data=None, seed_data=None, n_query=None, alpha=1.0, kernel=None
    ):
        """
        Args:
            candidate_data (pandas.DataFrame): data about the candidates to search over. Must have a "target" column,
                    and at least one additional column that can be used as descriptors.
            seed_data (pandas.DataFrame):  data which to fit the Agent to.
            n_query (int): number of queries in allowed. Defaults to 1.
            alpha (float): mixing parameter for uncertainties in UCB. Defaults to 1.0.
            kernel (scikit-learn kernel): Kernel object for the GP. Defaults to ConstantKernel(1.0)*RBF(1.0).
                    See scikit-learn.gaussian_process.kernels for details.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.cv_score = np.inf
        self.alpha = alpha
        self.kernel = kernel if kernel else ConstantKernel(1.0) * RBF(1.0)
        super(GenericGPUCB).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.candidate_data = candidate_data.drop(columns=["target"], axis=1)
        self.seed_data = seed_data
        X_seed = seed_data.drop(columns=["target"], axis=1)
        y_seed = seed_data["target"]
        steps = [
            ("scaler", StandardScaler()),
            (
                "GP",
                GaussianProcessRegressor(
                    kernel=self.kernel, normalize_y=True, n_restarts_optimizer=25
                ),
            ),
        ]
        self.pipeline = Pipeline(steps)
        self.cv_score = np.mean(
            -1
            * cross_val_score(self.pipeline, X_seed, y_seed, cv=KFold(3, shuffle=True))
        )
        self.pipeline.fit(X_seed, y_seed)
        t_pred, unc = self.pipeline.predict(self.candidate_data, return_std=True)
        t_pred += unc * self.alpha
        selected = np.argsort(-1.0 * t_pred)[: self.n_query]
        return candidate_data.loc[selected]


class GPflowBatchUCB(HypothesisAgent):
    """
    Generic Gaussian Process (GP) and upper confidence bound (UCB)
    based agent that tries to maximize a target that can
    be used in a batch-mode Bayesian Optimization setting.
    """

    def __init__(
            self, candidate_data=None, seed_data=None, n_query=None, alpha=1.0, kernel=None
    ):
        """
        Args:
            candidate_data (pandas.DataFrame): data about the candidates to search over. Must have a "target" column,
                    and at least one additional column that can be used as descriptors.
            seed_data (pandas.DataFrame):  data which to fit the Agent to.
            n_query (int): number of queries in allowed. Defaults to 1.
            alpha (float): mixing parameter for uncertainties in UCB. Defaults to 1.0.
            kernel (scikit-learn kernel): Kernel object for the GP. Defaults to ConstantKernel(1.0)*RBF(1.0).
                    See scikit-learn.gaussian_process.kernels for details.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.cv_score = np.inf
        self.alpha = alpha
        self.kernel = kernel if kernel else gpflow.kernels.RBF()
        super(GPflowBatchUCB).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        self.candidate_data = candidate_data.drop(columns=["target"], axis=1)
        self.seed_data = seed_data

        X_seed = seed_data.drop(columns=["target"], axis=1)
        y_seed = seed_data["target"].to_numpy().reshape(-1, 1)

        y_m, y_std = np.mean(y_seed), np.std(y_seed)
        y_seed = (y_seed - y_m) / y_std

        scaler = StandardScaler()
        scaler.fit(X_seed)

        r_seed, r_y, r_candidates = X_seed, y_seed, self.candidate_data

        selected = []
        for i in range(min(self.n_query, len(self.candidate_data))):
            x = scaler.transform(r_seed).astype(np.float64)
            y = r_y.astype(np.float64).reshape(-1, 1)

            m = gpflow.models.GPR((x, y), kernel=self.kernel)
            m.likelihood.variance.assign(0.1)

            opt = gpflow.optimizers.Scipy()
            opt.minimize(m.training_loss,
                                    m.trainable_variables, options=dict(maxiter=200))

            self.kernel = m.kernel
            pred, var = m.predict_f(scaler.transform(r_candidates.to_numpy().astype(np.float64)))
            t_pred = pred * y_std + y_m
            unc = np.sqrt(var) * y_std + y_m

            t_pred += unc * self.alpha
            s = np.argmax(t_pred)

            name = r_candidates.index.tolist()[s]
            print(s, (pred[s]*y_std+y_m)[0], t_pred[s][0], unc[s], candidate_data['target'].loc[name])
            selected.append(name)
            print(selected, candidate_data.loc[selected]['target'])
            r_seed = r_seed.append(r_candidates.loc[name])
            r_y = np.append(r_y, np.array([pred[s]]).reshape(1, 1), axis=0)
            r_candidates = r_candidates.drop(name)
        gpflow.utilities.reset_cache_bijectors(m)
        return self.candidate_data.loc[selected]