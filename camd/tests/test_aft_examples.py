import unittest
import os
import tempfile
import shutil
import pandas as pd

from sklearn.neural_network import MLPRegressor
from camd.agents import AgentRandom, AgentStabilityQBC
from camd.analysis import AnalyzeStability
from camd.utils import aft_loop, sync_s3_objs
from camd import S3_CACHE, CAMD_TEST_FILES


CAMD_LONG_TESTS = os.environ.get("CAMD_LONG_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_LONG_TESTS to run long tests"


# TODO: s3 sync doesn't work on jenkins
# @unittest.skipUnless(CAMD_LONG_TESTS, SKIP_MSG)
class AftLoopTestLong(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sync s3 objects required for this test
        sync_s3_objs()

    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_random_agent_loop(self):
        df = pd.read_csv(os.path.join(S3_CACHE, 'oqmd_voro_March25_v2.csv'))
        df_sub = df[df['N_species'] == 2].sample(frac=0.2)  # Downsampling candidates to 20% just for testing!
        n_seed = 5000  # Starting sample size
        n_query = 200  # his many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = AgentRandom
        agent_params = {'hull_distance': 0.05}  # Distance to hull to consider a finding as discovery (eV/atom)
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        for _ in range(6):
            aft_loop(self.tempdir, df, df_sub, n_seed, n_query, agent,
                     agent_params, analyzer, analyzer_params)
            self.assertTrue(True)

    def test_qbc_agent_loop(self):
        pass


class AftLoopTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_random_agent_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] == 2].sample(frac=0.5)  # Downsampling candidates to 20% just for testing!
        n_seed = 200  # Starting sample size
        n_query = 10  # his many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = AgentRandom
        agent_params = {'hull_distance': 0.05}  # Distance to hull to consider a finding as discovery (eV/atom)
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        for _ in range(6):
            aft_loop(self.tempdir, df, df_sub, n_seed, n_query, agent,
                     agent_params, analyzer, analyzer_params)
            self.assertTrue(True)

    def test_qbc_agent_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        # TODO: @murat can you check this, is it okay to do?
        df_sub = df[df['N_species'] <= 3]# .sample(frac=0.8) # Downsampling candidates to 20% just for testing!
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = AgentStabilityQBC
        agent_params = {
            'ML_algorithm': MLPRegressor,
            'ML_algorithm_params': {'hidden_layer_sizes': (84, 50)},
            'N_members': 10,  # Committee size
            'hull_distance': 0.05  # Distance to hull to consider a finding as discovery (eV/atom)
            }
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}

        for _ in range(6):
            aft_loop(self.tempdir, df, df_sub, n_seed, n_query, agent,
                     agent_params, analyzer, analyzer_params)


if __name__ == '__main__':
    unittest.main()
