#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import boto3
import os
import pandas as pd

os.environ['CAMD_S3_BUCKET'] = 'camd-test'
from camd.campaigns.structure_discovery import ProtoDFTCampaign
from camd.agent.stability import AgentStabilityML5, AgentStabilityAdaBoost
from camd.analysis import StabilityAnalyzer
from camd.experiment.base import ATFSampler
from camd import CAMD_S3_BUCKET, CAMD_TEST_FILES
from camd.utils.data import filter_dataframe_by_composition, load_dataframe
from monty.tempfile import ScratchDir
from sklearn.neural_network import MLPRegressor

CAMD_DFT_TESTS = os.environ.get("CAMD_DFT_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_DFT_TESTS to run long tests"


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="proto-dft-2/runs/Si").delete()
    bucket.objects.filter(Prefix="proto-dft-2/submit/Si").delete()


class ProtoDFTCampaignTest(unittest.TestCase):
    def test_simulated(self):
        # Note that there's a small issue with pickled results here that may not have
        # certain spin and magnetization flags set - pickled Vasprun objects may not
        # be completely compatible with latest version of pymatgen
        exp_dataframe = pd.read_pickle(os.path.join(CAMD_TEST_FILES, "mn-ni-o-sb.pickle"))
        experiment = ATFSampler(exp_dataframe)
        candidate_data = exp_dataframe.iloc[:, :-11]
        # Set up agents and loop parameters
        agent = AgentStabilityAdaBoost(
            model=MLPRegressor(hidden_layer_sizes=(84, 50)),
            n_query=2,
            hull_distance=0.2,
            exploit_fraction=1.0,
            uncertainty=True,
            alpha=0.5,
            diversify=True,
            n_estimators=20
        )
        analyzer = StabilityAnalyzer(hull_distance=0.2)
        # Reduce seed_data
        icsd_data = load_dataframe("oqmd1.2_exp_based_entries_featurized_v2")
        seed_data = filter_dataframe_by_composition(icsd_data, "MnNiOSb")
        leftover = ~icsd_data.index.isin(seed_data.index)
        # Add some random other data to test compositional flexibility
        seed_data = seed_data.append(icsd_data.loc[leftover].sample(30))
        del icsd_data
        with ScratchDir('.'):
            campaign = ProtoDFTCampaign(
                candidate_data=candidate_data, agent=agent, experiment=experiment,
                analyzer=analyzer, seed_data=seed_data,
                heuristic_stopper=5
            )
            campaign.autorun()
            self.assertTrue(os.path.isfile('hull_finalized.png'))

    @unittest.skipUnless(CAMD_DFT_TESTS, SKIP_MSG)
    def test_cached_campaign(self):
        with ScratchDir('.'):
            campaign = ProtoDFTCampaign.from_chemsys_high_quality("Si")
            # Test seed data has other data
            self.assertGreater(len(campaign.seed_data), 36581)

    @unittest.skipUnless(CAMD_DFT_TESTS, SKIP_MSG)
    def test_simple_dft(self):
        with ScratchDir('.'):
            campaign = ProtoDFTCampaign.from_chemsys("Si")
            # Nerf agent a bit
            agent = AgentStabilityML5(n_query=2)
            campaign.agent = agent
            campaign.autorun()
            teardown_s3()


if __name__ == '__main__':
    unittest.main()
