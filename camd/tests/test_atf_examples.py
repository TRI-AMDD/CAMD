import unittest
import os
import tempfile
import shutil
import pandas as pd

from sklearn.neural_network import MLPRegressor
from camd.agent.agents import QBCStabilityAgent, GaussianProcessStabilityAgent, SVGProcessStabilityAgent, \
    BaggedGaussianProcessStabilityAgent, AgentStabilityAdaBoost
from camd.agent.base import RandomAgent
from camd.analysis import AnalyzeStability as AnalyzeStability
from camd.experiment import ATFSampler
from camd.loop import Loop
from camd.utils.s3 import cache_s3_objs
from camd import S3_CACHE, CAMD_TEST_FILES
from camd.utils.data import load_default_atf_data


CAMD_LONG_TESTS = os.environ.get("CAMD_LONG_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_LONG_TESTS to run long tests"


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
        df = load_default_atf_data()
        n_seed = 5000
        agent = RandomAgent(n_query=200)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df)
        candidate_data = df

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)

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
        df = load_default_atf_data()
        n_seed = 200  # Starting sample size
        agent = RandomAgent(n_query=10)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df)
        candidate_data = df
        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)

        new_loop.initialize()
        self.assertFalse(new_loop.create_seed)

        for _ in range(6):
            new_loop.run()
            self.assertTrue(True)

        # Testing the continuation
        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        self.assertTrue(new_loop.initialized)
        self.assertEqual(new_loop.iteration, 6)
        self.assertEqual(new_loop.loop_state, None)

        new_loop.run()
        self.assertTrue(True)
        self.assertEqual(new_loop.iteration, 7)

    def test_qbc_agent_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = QBCStabilityAgent(model=MLPRegressor(hidden_layer_sizes=(84, 50)),
                                  n_query=10, hull_distance=0.05, alpha=0.5)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df_sub)
        candidate_data = df_sub
        path = '.'

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        new_loop.initialize()
        self.assertTrue(new_loop.initialized)

        new_loop.auto_loop(6)
        self.assertTrue(True)

    def test_simple_gp_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = GaussianProcessStabilityAgent(n_query=n_query, hull_distance=0.05, alpha=0.5)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df_sub)
        candidate_data = df_sub

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        new_loop.initialize()
        self.assertTrue(new_loop.initialized)

        new_loop.auto_loop(6)
        self.assertTrue(True)

    def test_gp_bagging(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        agent = BaggedGaussianProcessStabilityAgent(
            n_query=10,
            hull_distance=0.05,
            alpha=0.5, # Fraction of std to include in expected improvement
            n_estimators=2,
            max_samples=195
        )
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(df_sub)
        candidate_data = df_sub

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        new_loop.initialize()
        self.assertTrue(new_loop.initialized)

        new_loop.auto_loop(6)
        self.assertTrue(True)

    def test_adaboost_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        n_query = 10  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = AgentStabilityAdaBoost(model=MLPRegressor(hidden_layer_sizes=(84, 50)),
                                       n_query=10, exploit_fraction=1.0, alpha=0.5,
                                       n_estimators=10)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(df_sub)
        candidate_data = df_sub

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        new_loop.initialize()
        self.assertTrue(new_loop.initialized)

        new_loop.auto_loop(6)
        self.assertTrue(True)

    def test_mp_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df_analysis.csv'),)
        df['id'] = [int(mp_id.replace("mp-", "").replace('mvc-', ''))
                    for mp_id in df['id']]
        df.set_index("id")
        df['Composition'] = df['formula']

        # Just use the Ti-O-N chemsys
        seed_data = df.iloc[:38]
        candidate_data = df.iloc[38:209]
        n_query = 20  # This many new candidates are "calculated with DFT" (i.e. requested from Oracle -- DFT)
        agent = RandomAgent(n_query=20)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(dataframe=df)
        # candidate_data = df
        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        seed_data=seed_data)

        new_loop.initialize()
        self.assertFalse(new_loop.create_seed)

        for iteration in range(6):
            new_loop.run()
            self.assertTrue(
                os.path.isfile("hull_{}.png".format(iteration)))
            if iteration >= 1:
                self.assertTrue(
                    os.path.isfile("report.png"))

        # Testing the continuation
        new_loop = Loop(candidate_data, agent, experiment, analyzer)
        self.assertTrue(new_loop.initialized)
        self.assertEqual(new_loop.iteration, 6)
        self.assertEqual(new_loop.loop_state, None)

        new_loop.run()
        self.assertTrue(True)
        self.assertEqual(new_loop.iteration, 7)


@unittest.skipUnless(CAMD_LONG_TESTS, SKIP_MSG)
class AtfSVGPLoopTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_svgp_loop(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, 'test_df.csv'))
        df_sub = df[df['N_species'] <= 3]
        n_seed = 200  # Starting sample size
        agent = SVGProcessStabilityAgent(n_query=10, hull_distance=0.05, alpha=0.5, M=100)
        analyzer = AnalyzeStability(hull_distance=0.05)
        experiment = ATFSampler(df_sub)
        candidate_data = df_sub

        new_loop = Loop(candidate_data, agent, experiment, analyzer,
                        create_seed=n_seed)
        new_loop.initialize()
        self.assertTrue(new_loop.initialized)

        new_loop.auto_loop(3)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
