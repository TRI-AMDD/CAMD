# Copyright Toyota Research Institute 2019
"""
Contains generic agents which should not be constrained
to a particular mode of materials discovery or associated
decision-making logic
"""
import numpy as np
import GPy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LinearRegression

from camd.agent.base import HypothesisAgent


class GenericGPUCB(HypothesisAgent):
    """
    Generic Gaussian Process (GP) and upper confidence bound (UCB)
    based agent that tries to maximize a target. This provide a "naive" batch mode BO.
    We strongly recommend using GPBatchUCB instead.
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
        """
        Get hypotheses method for GenericGPUCB agent

        Args:
            candidate_data (pandas.DataFrame): dataframe of candidates
            seed_data (pandas.DataFrame): dataframe of prior data on
                which to fit GPUCB

        Returns:
            (pandas.DataFrame): top candidates from the GPUCB algorithm

        """

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
        return candidate_data.iloc[selected]


class GPBatchUCB(HypothesisAgent):
    """
    Generic Gaussian Process (GP) and upper confidence bound (UCB)
    based agent that tries to maximize a target that can
    be used in a batch-mode Bayesian Optimization setting.
    This class implements a version of the Batch-UCB algorithm described by
    Desautels et al. Journal of Machine Learning Research 15 (2014) 4053-4103
    which shares the same basis with Srinivas et al. arXiv preprint arXiv:0912.3995 (2009).
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query=None,
        mode="batch",
        alpha=1.0,
        kernel=None,
        **kwargs
    ):
        """
        Args:
            candidate_data (pandas.DataFrame): data about the candidates to search over. Must have a "target" column,
                    and at least one additional column that can be used as descriptors.
            seed_data (pandas.DataFrame):  data which to fit the Agent to.
            n_query (int): number of queries in allowed. Defaults to 1.
            mode (str): "batch" or "naive"; corresponding to original BUCB algorithm (where batches
                composed iteratively) and a naive batch algorithm (where batch is composed in one-shot
                based on ranking). "naive" is mostly for benchmarking, but is also faster so might be
                useful large, limiting dataset sizes.
            alpha (float or str): mixing parameter for uncertainties in UCB. If a float is given, agent will
                use the same constant alpha throughout the campaign. Defaults to 1.0. Setting this as 'auto' will
                use the Theorem 1 from Srivanasan et al. to determine the alpha during batch composition.
                'auto' has two parameters, 'delta' and 'premultip', which default to 0.1 and 0.05 respectively,
                but can be modified by passing as kwargs to the agent. if mode is "naive" this can only be a flaot.
            kernel (GPy kernel): Kernel object for the GP. Defaults to RBF.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.mode = mode
        self.alpha = alpha
        self.kernel = kernel
        self.kwargs = kwargs
        self.cv_score = np.inf

        super(GPBatchUCB).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Methods for getting hypotheses according to the GPBatchUCB algorithm

        Args:
            candidate_data (pandas.DataFrame): candidate data
            seed_data (pandas.DataFrame): seed data

        Returns:
            (pandas.DataFrame): selected hypotheses

        """
        self.candidate_data = candidate_data.drop(
            columns=["target"], axis=1, errors="ignore"
        )

        if seed_data is not None:
            self.seed_data = seed_data
        else:
            raise ValueError(
                "GPBatchUCB Agent requires a finite seed as input. "
                "If you are using this as part of a Campaign, consider "
                "the create_seed option."
            )

        fb_start = max(len(self.seed_data), 1)

        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=self.candidate_data.shape[1])

        X_seed = self.seed_data.drop(columns=["target"], axis=1)
        y_seed = self.seed_data["target"].to_numpy().reshape(-1, 1)

        y_m, y_std = np.mean(y_seed), np.std(y_seed)
        y_seed = (y_seed - y_m) / y_std

        scaler = StandardScaler()
        scaler.fit(X_seed)

        r_seed, r_y, r_candidates = X_seed, y_seed, self.candidate_data

        if self.mode == "batch":
            batch = []
            for i in range(min(self.n_query, len(self.candidate_data))):
                x = scaler.transform(r_seed).astype(np.float64)
                y = r_y.astype(np.float64).reshape(-1, 1)

                m = GPy.models.GPRegression(
                    x,
                    y,
                    kernel=self.kernel,
                    noise_var=self.kwargs.get("noise_var", 1.0),
                )
                m.optimize(
                    optimizer=self.kwargs.get("optimizer", "bfgs"),
                    max_iters=self.kwargs.get("max_iters", 1000),
                )
                self.kernel = m.kern

                y_pred, var = m.predict(
                    scaler.transform(r_candidates.to_numpy().astype(np.float64))
                )
                t_pred = y_pred * y_std + y_m
                unc = np.sqrt(var) * y_std + y_m

                if self.alpha == "auto":
                    _t = i + fb_start
                    alpha = self.kwargs.get("premultip", 0.05) * np.sqrt(
                        2
                        * np.log(
                            len(self.candidate_data)
                            * _t ** 2
                            * np.pi ** 2
                            / 6
                            / self.kwargs.get("delta", 0.1)
                        )
                    )
                    print("- alpha.{}: ".format(i), alpha)
                else:
                    alpha = self.alpha

                t_pred += unc * alpha
                s = np.argmax(t_pred)

                name = r_candidates.index.tolist()[s]
                batch.append(name)
                r_seed = r_seed.append(r_candidates.loc[name])
                r_y = np.append(r_y, np.array([y_pred[s]]).reshape(1, 1), axis=0)
                r_candidates = r_candidates.drop(name)

        elif self.mode == "naive":
            x = scaler.transform(r_seed).astype(np.float64)
            y = r_y.astype(np.float64).reshape(-1, 1)
            m = GPy.models.GPRegression(
                x, y, kernel=self.kernel, noise_var=self.kwargs.get("noise_var", 1.0)
            )
            self.kernel = m.kern
            m.optimize(
                optimizer=self.kwargs.get("optimizer", "bfgs"),
                max_iters=self.kwargs.get("max_iters", 1000),
            )
            y_pred, var = m.predict(
                scaler.transform(r_candidates.to_numpy().astype(np.float64))
            )
            t_pred = y_pred * y_std + y_m
            unc = np.sqrt(var) * y_std + y_m
            t_pred += unc * self.alpha
            indices = np.argsort(-1.0 * t_pred.flatten())[: self.n_query]
            batch = r_candidates.index.to_numpy()[indices].tolist()
        else:
            raise NotImplementedError("Unknown mode for GPBatchUCB Agent")

        return self.candidate_data.loc[batch]


class LinearAgent(HypothesisAgent):
    """
    Linear regression based agent that tries to maximize a target.
    Best for simple checks and benchmarks.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        n_query: int = None,
        fit_intercept: bool = True,
        positive: bool = False,
    ):

        """
        Args:
            candidate_data (pandas.DataFrame): data about the candidates to search over. Must have a "target" column,
                    and at least one additional column that can be used as descriptors.
            seed_data (pandas.DataFrame):  data which to fit the Agent to.
            n_query (int): number of queries in allowed. Defaults to 1.
            fit_intercept (bool): if the intercept is fit for the linear regression
            positive (bool): if true, constraint coefficients to be positive for the linear regression
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        self.fit_intercept = fit_intercept
        self.positive = positive
        super(LinearAgent).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None):
        """
        Methods for getting hypotheses using linear regression

        Args:
            candidate_data (pandas.DataFrame): candidate data
            seed_data (pandas.DataFrame): seed data

        Returns:
            (pandas.DataFrame): selected hypotheses

        """
        # Fit on known data
        self.candidate_data = candidate_data.drop(
            columns=["target"], axis=1, errors="ignore"
        )

        if seed_data is not None:
            self.seed_data = seed_data
        else:
            raise ValueError(
                "Linear Agent requires a finite seed as input. "
                "If you are using this as part of a Campaign, consider "
                "the create_seed option."
            )

        X_seed = seed_data.drop(columns=["target"], axis=1)
        y_seed = seed_data["target"]
        steps = [
            ("scaler", StandardScaler()),
            (
                "linear",
                LinearRegression(),
            ),
        ]
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X_seed, y_seed)
        output = self.pipeline.predict(self.candidate_data)
        sorted_output = np.argsort(output)[::-1]
        selected = sorted_output[: self.n_query]
        return candidate_data.iloc[selected]
