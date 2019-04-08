import unittest
import os
import tempfile
import shutil
import pandas as pd

from camd.agents import AgentRandom, AgentStabilityQBC
from camd.analysis import AnalyzeStability

from camd.utils import aft_loop


# TODO: remove skips when files are provided
class AftLoopTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    @unittest.skip
    def test_random_agent_loop(self):
        df = pd.read_csv('../oqmd_voro_March25_v2.csv')
        df_sub = df[df['N_species'] == 2].sample(frac=0.2)  # Downsampling candidates to 20% just for testing!
        n_seed = 5000  # Starting sample size
        n_query = 200  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = AgentRandom
        agent_params = {'hull_distance': 0.05}  # Distance to hull to consider a finding as discovery (eV/atom)
        analyzer = AnalyzeStability
        analyzer_params = {'hull_distance': 0.05}
        for _ in range(6):
            aft_loop(self.tempdir, df, df_sub, n_seed, n_query, agent,
                     agent_params, analyzer, analyzer_params)
            self.assertTrue(True)

    @unittest.skip
    def test_qbc_agent_loop(self):
        pass


if __name__ == '__main__':
    unittest.main()
