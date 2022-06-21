import pandas as pd
import numpy as np
import tensorflow as tf

from m3gnet.models import M3GNet, Potential, Relaxer
from m3gnet.trainers import PotentialTrainer

from camd.agent.base import HypothesisAgent

class M3GNetAgent(HypothesisAgent):
    def __init__(self, m3gnet=None, candidate_data=None, seed_data=None, n_query=None):
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.n_query = n_query if n_query else 1
        # self.cv_score = np.inf
        if not m3gnet:
            self.m3gnet = M3GNet(is_intensive=False)
        self.potential = Potential(model=m3gnet)
        self.trainer = PotentialTrainer(
                potential=self.potential, optimizer=tf.keras.optimizers.Adam(1e-3)
            )
        super(M3GNetAgent).__init__()
        
    def get_hypotheses(self, candidate_data, seed_data=None, retrain=False):
        if 'target' in candidate_data.columns:
            self.candidate_data = candidate_data.drop(columns=["target"], axis=1)
        else:
            self.candidate_data = candidate_data
        self.seed_data = seed_data
        X_seed = seed_data.drop(columns=["target"], axis=1, errors='ignore')
        y_seed = seed_data["target"]
        if retrain and 'calcs_reversed' in X_seed.columns:
            self.train(X_seed)
        relaxer = Relaxer(potential=self.trainer.potential)
        t_pred = []
        for s in self.candidate_data['structure']:
            t = relaxer.relax(s)['trajectory']
            e = t.energies[-1].flatten()[0]
            t_pred.append(e)
        selected = np.argsort(-1.0 * np.array(t_pred))[: self.n_query]
        return candidate_data.iloc[selected]
    
    def train(self, train_data, epochs=10, kwargs={}):
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
        X_struct, Xe, Xf, Xs = self.reverse_calcs(train_data['calcs_reversed'])
        self.trainer.train(graphs_or_structures = X_struct,
                           energies = Xe,
                           forces = Xf,
                           stresses = Xs,
                           epochs = epochs,
                           **kwargs)

    def reverse_calcs(self, retrain_data, data_screening=None):
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
                for step, ionic_steps in enumerate(task['output']['ionic_steps']):
                # entry = {"_id": d["_id"]['oid']}
                # entry.update({"energy": ionic_steps['e_fr_energy'],
                             # "force": ionic_steps['forces'],
                             # "stress": ionic_steps['stress'],
                             # "structure": ionic_steps['structure'],
                             # "ionic_step": step,
                             # "task": relax["task"]["type"]})
                #TODO: implement data selection criteria
                    X_struct.append(ionic_steps['structure'])
                    Xe.append(ionic_steps['e_fr_energy'] / ionic_steps['structure'].num_sites)
                    Xf.append(ionic_steps['forces'])
                    Xs.append(ionic_steps['stress'])
        return X_struct, Xe, Xf, Xs
