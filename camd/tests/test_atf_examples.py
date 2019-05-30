import unittest
import os
import tempfile
import shutil
import pandas as pd

from sklearn.neural_network import MLPRegressor
from camd.agent.agents import QBCStabilityAgent
from camd.agent.base import RandomAgent
from camd.analysis import AnalyzeStability
from camd.experiment import ATFSampler
from camd.loop import Loop
from camd.utils.s3 import cache_s3_objs
from camd import S3_CACHE, CAMD_TEST_FILES


CAMD_LONG_TESTS = os.environ.get("CAMD_LONG_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_LONG_TESTS to run long tests"


# TODO: s3 sync doesn't currently work on jenkins
@unittest.skipUnless(CAMD_LONG_TESTS, SKIP_MSG)
class AftLoopTestLong(unittest.TestCase):
    def setUp(self):
        # Normally put in setUpClass but that doesn't
        # work with skip apparently
        cache_s3_objs('camd/shared-data/oqmd_voro_March25_v2.csv')
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_random_agent_loop(self):

        self.assertTrue(os.path.exists(os.path.join(S3_CACHE, 'camd/shared-data/oqmd_voro_March25_v2.csv')))

        df = pd.read_csv(os.path.join(S3_CACHE, 'camd/shared-data/oqmd_voro_March25_v2.csv')).sample(frac=0.2)
        N_seed = 5000
        N_query = 200
        agent = RandomAgent
        agent_params = {'hull_distance': 0.05, 'N_query': N_query,
                        'N_species': 2}
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        experiment = ATFSampler
        experiment_params = {'dataframe': df}
        candidate_data = df
        path = '.'

        new_loop = Loop(path, candidate_data, agent, experiment, analyzer,
                        agent_params=agent_params, analyzer_params=analyzer_params, experiment_params=experiment_params,
                        create_seed=N_seed)

        new_loop.initialize()
        self.assertFalse(new_loop.create_seed)

        for _ in range(6):
            new_loop.run()
            self.assertTrue(True)

    def test_qbc_agent_loop(self):
        pass


class AtfLoopTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_random_agent_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv')).sample(frac=0.5)
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = RandomAgent
        agent_params = {'hull_distance': 0.05, 'N_query': n_query,
                        'N_species': 2}  # Distance to hull to consider a finding as discovery (eV/atom)
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        experiment = ATFSampler
        experiment_params = {'dataframe': df}
        candidate_data = df
        path = '.'
        new_loop = Loop(path, candidate_data, agent, experiment, analyzer,
                        agent_params=agent_params, analyzer_params=analyzer_params,
                        experiment_params=experiment_params,
                        create_seed=n_seed)

        new_loop.initialize()
        self.assertFalse(new_loop.create_seed)

        for _ in range(6):
            new_loop.run()
            self.assertTrue(True)

    def test_qbc_agent_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = QBCStabilityAgent
        agent_params = {
            'ML_algorithm': MLPRegressor,
            'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
            'N_query': n_query,
            'N_members': 10,  # Committee size
            'hull_distance': 0.05,  # Distance to hull to consider a finding as discovery (eV/atom)
            'frac': 0.5  # Fraction to exploit (rest will be explored -- randomly picked)
        }
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        experiment = ATFSampler
        experiment_params = {'dataframe': df_sub}
        candidate_data = df_sub
        path = '.'

        new_loop = Loop(path, candidate_data, agent, experiment, analyzer,
                        agent_params=agent_params, analyzer_params=analyzer_params,
                        experiment_params=experiment_params,
                        create_seed=n_seed)
        new_loop.auto_loop(6)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
