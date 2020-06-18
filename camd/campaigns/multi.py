"""
Prototype multi-fidelity campaign
"""

# Import packages
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Set pandas view options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from matminer.datasets.dataset_retrieval import load_dataset
## Generate magpie features
from matminer.featurizers.composition import ElementProperty
from pymatgen import Composition
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pymatgen import MPRester, Composition
mpr = MPRester('68rWEneaZFyIaKh15uKr') # provide your API key here or add it to pymatgen

# Load dataset
# ---------------------------------------------------------------------
# featurized_data = pd.read_csv('featurized_brgoch_data.csv', index_col=0)
# print(featurized_data.head(2))


# Agent Design
# ----------------------------------------------------------------------



# Experiment
# -----------------------------------------------------------------------



# Analyzer
# -----------------------------------------------------------------------
class MultiAnalyzer():
    def __init__(self, target_property, property_range):
        self.target_property = target_property 
        self.property_range = property_range
        
    def _filter_df_by_property_range(self, df):
        """
        Helper function to filter df by property range
        
        Args:
            df (DataFrame): dataframe to be filtered
        """
        return df[(df[self.target_property] >= self.property_range[0]) & 
                  (df[self.target_property] <= self.property_range[1])]
    
    def analyze(self, new_experimental_results, seed_data):
        # new discovery during the current iteration
        new_discovery = new_experimental_results[new_experimental_results['expt_calculated'] == 1]
        new_discovery = self._filter_df_by_property_range(new_discovery)
        # Uncertainty?
        
        # total discovery up to (& including) the current iteration
        new_seed = seed_data.append(new_experimental_results)
        total_discovery = new_seed[new_seed['expt_calculated'] == 1]
        total_discovery = self._filter_df_by_property_range(total_discovery)
        
        summary = pd.DataFrame(
            {
                "discovery_rate": [len(new_discovery) / len(new_experimental_results)],
                "total_discovery_frac": [len(total_discovery) / len(new_seed)]
                # "discoveries_per_cost": [len(new_discovery) / cost
            }
        )
        return summary, new_seed
 