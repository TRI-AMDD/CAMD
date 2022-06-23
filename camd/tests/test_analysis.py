#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import os
import pandas as pd
import pickle
from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES
from camd.analysis import StabilityAnalyzer, AnalyzeStructures, GenericATFAnalyzer
from camd.experiment.base import ATFSampler
from camd.agent.base import RandomAgent
from camd.campaigns.base import Campaign
from camd.utils.data import filter_dataframe_by_composition


class StabilityAnalyzerTest(unittest.TestCase):
    def test_plot_hull(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = df['formula']

        # Test 2D
        with ScratchDir('.'):
            analyzer = StabilityAnalyzer(hull_distance=0.1)
            filtered = filter_dataframe_by_composition(df, "TiO")
            analyzer.plot_hull(filtered, new_result_ids=["mp-685151", "mp-755875"],
                               filename="hull.png")
            self.assertTrue(os.path.isfile("hull.png"))

        # Test 3D
        with ScratchDir('.'):
            analyzer.hull_distance = 0.05
            filtered = filter_dataframe_by_composition(df, "TiNO")
            analyzer.plot_hull(filtered, new_result_ids=["mp-776280", "mp-30998"],
                               filename="hull.png")
            self.assertTrue(os.path.isfile("hull.png"))

    def test_analyze(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = df['formula']
        analyzer = StabilityAnalyzer(hull_distance=0.1)
        seed_data = filter_dataframe_by_composition(df, "TiNO")
        # TODO: resolve drop_duplicates filtering mp data
        seed_data = seed_data.drop_duplicates(keep='last').dropna()
        new_exp_indices = ["mp-30998", "mp-572822"]
        new_experimental_results = seed_data.loc[new_exp_indices]
        seed_data = seed_data.drop(index=new_exp_indices)
        agent = RandomAgent()
        exp = ATFSampler(df)
        campaign = Campaign(
            candidate_data=new_experimental_results,
            agent=agent, experiment=exp,
        )
        exp.submit(new_experimental_results)
        summary1 = analyzer.analyze(campaign)
        campaign.seed_data = seed_data
        summary2 = analyzer.analyze(campaign)
        self.assertTrue(summary1.loc[0, 'new_discovery'], 2)
        self.assertTrue(summary2.loc[0, 'new_discovery'], 1)


class StructureAnalyzerTest(unittest.TestCase):
    def test_analyze_vaspqmpy_jobs(self):
        jobs = pd.read_pickle(os.path.join(CAMD_TEST_FILES, "mn-ni-o-sb.pickle"))
        analyzer = AnalyzeStructures()
        self.assertEqual(analyzer.analyze_vaspqmpy_jobs(jobs, against_icsd=True, use_energies=False),
                         [True, True, True])

        self.assertEqual(analyzer.analyze_vaspqmpy_jobs(jobs, against_icsd=True, use_energies=True),
                         [True, True, True])


class GenericATFAnalyzerTest(unittest.TestCase):
    def test_analyze(self):
        seed_size = 10
        exploration_df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_ATF.csv"))
        record = pickle.load(open(os.path.join(CAMD_TEST_FILES, "seed_data_ATF.pickle"), "rb"))
        analyzer = GenericATFAnalyzer(percentile=0.01)
        agent = RandomAgent()
        exp = ATFSampler(exploration_df)
        campaign = Campaign(candidate_data=exploration_df, agent=agent, experiment=exp)
        # Submit best ones, ALM should be 1, True, 1 respectively
        exp.submit(exploration_df.loc[[1679, 1737]])
        summary = analyzer.analyze(campaign)

        self.assertEqual(tuple(summary['deALM'].loc[0]), (1, 1))
        self.assertEqual(summary['anyALM'].loc[0], True)
        self.assertAlmostEqual(summary['allALM'].loc[0], 2 / 19)
        # self.assertEqual(summary['simALM'].to_list()[0].shape[0], record.shape[0]-seed_size)


if __name__ == '__main__':
    unittest.main()
