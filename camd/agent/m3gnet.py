"""This module defines agents and various functions
that use megnet for candidate selection or candidate filtration
"""
import numpy as np
import tensorflow as tf
from multiprocessing import cpu_count

from m3gnet.models import M3GNet, Potential, Relaxer
from m3gnet.trainers import PotentialTrainer

from camd.agent.stability import StabilityAgent
from camd.agent.base import HypothesisAgent


def reverse_calcs(retrain_data, data_screening=None):
    """
    Prepare the data for training
    Args:
        retrain_data (pd.DataFrame):
        data_screening (dict): retrain data select criteria
                            example: {"task": "relax1",
                                    "ionic_step": "last_n"}
    Returns: lists
        X_struct, Xe, Xf, Xs
    """
    X_struct = []
    Xe = []
    Xf = []
    Xs = []
    for d in retrain_data:
        for task in d:
            for step, ionic_steps in enumerate(task["output"]["ionic_steps"]):
                # TODO: implement data selection criteria
                X_struct.append(ionic_steps["structure"])
                Xe.append(
                    ionic_steps["e_fr_energy"] / ionic_steps["structure"].num_sites
                )
                Xf.append(ionic_steps["forces"])
                Xs.append(ionic_steps["stress"])
    return X_struct, Xe, Xf, Xs


class M3GNetStabilityAgent(StabilityAgent):
    """Stability agent that uses megnet for estimation of energy"""

    def __init__(
        self,
        m3gnet=None,
        learning_rate=1e-3,
        candidate_data=None,
        seed_data=None,
        n_query=1,
        hull_distance=0.0,
        parallel=cpu_count(),
    ):
        """


        Args:
            m3gnet (M3gnet): m3gnet object, graph CNN from m3gnet package
                which determines energy, forces, stresses, and can do relaxation
                etc.
            learning_rate (float): learning rate for m3gnet fitting
            candidate_data (pd.DataFrame): candidate data to predict on
            seed_data (pd.DataFrame): seed data to fit on
            n_query (int): number of candidates to select
            hull_distance (float): hull distance to consider
            parallel (int): number of parallel processes to use for
                phase determination
        """
        super(M3GNetStabilityAgent, self).__init__(
            candidate_data=candidate_data,
            seed_data=seed_data,
            n_query=n_query,
            hull_distance=hull_distance,
            parallel=parallel,
        )
        if not m3gnet:
            self.m3gnet = M3GNet(is_intensive=False)
        else:
            self.m3gnet = m3gnet
        self.potential = Potential(model=self.m3gnet)
        self.trainer = PotentialTrainer(
            potential=self.potential, optimizer=tf.keras.optimizers.Adam(learning_rate)
        )

    def get_hypotheses(
        self, candidate_data, seed_data=None, retrain_committee=False, retrain_epochs=1
    ):
        """
        Get hypotheses method for downselecting candidates

        Args:
            candidate_data (pd.DataFrame): candidate data to predict on and select from
            seed_data (pd.DataFrame): seed data to fit on
            retrain_committee (bool): whether to retrain m3gnet on each run
            retrain_epochs (int): number of epochs for retraining

        Returns:
            (pd.DataFrame): selected candidates for experiment

        """
        if "target" in candidate_data.columns:
            self.candidate_data = candidate_data.drop(columns=["target"], axis=1)
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data
        X_seed = seed_data.drop(columns=["target"], axis=1, errors="ignore")
        # y_seed = seed_data["target"]
        if retrain_committee and "calcs_reversed" in X_seed.columns:
            self.train(X_seed, retrain_epochs)
        relaxer = Relaxer(potential=self.trainer.potential)
        t_pred = []
        for s in candidate_data["structure"]:
            t = relaxer.relax(s)["trajectory"]
            e = t.energies[-1].flatten()[0]
            t_pred.append(e)
        # TODO: get real formation energy instead of energy per atom
        self.update_candidate_stabilities(t_pred, sort=True, floor=-6.0)
        # Find the most stable ones up to n_query within hull_distance
        stability_filter = self.candidate_data["pred_stability"] <= self.hull_distance
        within_hull = self.candidate_data[stability_filter]
        return within_hull.head(self.n_query)

    def train(self, train_data, epochs=10, kwargs={}):
        """
        Train the potential
        Args:
            train_data (pd.DataFrame): data for re-train,
                                        same format as seed data
                                        should contain "calcs_reversed" column
            epochs (int): number of max epoch to train
            kwargs (dict): parameters for trainer.train

        Returns:
            None
        """
        X_struct, Xe, Xf, Xs = reverse_calcs(train_data["calcs_reversed"])
        self.trainer.train(
            graphs_or_structures=X_struct,
            energies=Xe,
            forces=Xf,
            stresses=Xs,
            epochs=epochs,
            **kwargs
        )


class M3GNetHypothesisAgent(HypothesisAgent):
    """Generic agent for AL using m3gnet"""

    def __init__(self, m3gnet=None, candidate_data=None, seed_data=None, n_query=None):

        """
        Args:
            m3gnet (M3gnet): m3gnet object, graph CNN from m3gnet package
                which determines energy, forces, stresses, and can do relaxation
                etc.
            candidate_data (pd.DataFrame): candidate data to predict on
            seed_data (pd.DataFrame): seed data to fit on
            n_query (int): number of candidates to select
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        # self.cv_score = np.inf
        if not m3gnet:
            self.m3gnet = M3GNet(is_intensive=False)
        else:
            self.m3gnet = m3gnet
        self.potential = Potential(model=self.m3gnet)
        self.trainer = PotentialTrainer(
            potential=self.potential, optimizer=tf.keras.optimizers.Adam(1e-3)
        )
        super(M3GNetHypothesisAgent).__init__()

    def get_hypotheses(self, candidate_data, seed_data=None, retrain=False):

        """
        Get hypotheses method for downselecting candidates

        Args:
            candidate_data (pd.DataFrame): candidate data to predict on and select from
            seed_data (pd.DataFrame): seed data to fit on
            retrain (bool): whether to retrain m3gnet on each run

        Returns:
            (pd.DataFrame): selected candidates for experiment

        """
        if "target" in candidate_data.columns:
            self.candidate_data = candidate_data.drop(columns=["target"], axis=1)
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data
        X_seed = seed_data.drop(columns=["target"], axis=1, errors="ignore")
        # y_seed = seed_data["target"]
        if retrain and "calcs_reversed" in X_seed.columns:
            self.train(X_seed)
        relaxer = Relaxer(potential=self.trainer.potential)
        t_pred = []
        for s in self.candidate_data["structure"]:
            t = relaxer.relax(s)["trajectory"]
            e = t.energies[-1].flatten()[0]
            t_pred.append(e)
        selected = np.argsort(-1.0 * np.array(t_pred))[: self.n_query]
        return candidate_data.iloc[selected]

    def train(self, train_data, epochs=10, kwargs=None):
        """
        Train the potential

        Args
            train_data (pd.DataFrame): data for re-train,
                                        same format as seed data
                                        should contain "calcs_reversed" column
            epochs (int): number of max epoch to train
            kwargs (dict): parameters for trainer.train
        Returns:
            None
        """
        if kwargs is None:
            kwargs = {}
        X_struct, Xe, Xf, Xs = reverse_calcs(train_data["calcs_reversed"])
        self.trainer.train(
            graphs_or_structures=X_struct,
            energies=Xe,
            forces=Xf,
            stresses=Xs,
            epochs=epochs,
            **kwargs
        )
