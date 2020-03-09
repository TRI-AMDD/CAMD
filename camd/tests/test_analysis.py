#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import os
import pandas as pd
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from camd import CAMD_TEST_FILES
from camd.analysis import StabilityAnalyzer, AnalyzeStructures


class AnalysisTest(unittest.TestCase):
    def test_stability_analyzer(self):
        pass

    def test_plot_hull(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = df['formula']

        # Test 2D
        with ScratchDir('.'):
            analyzer = StabilityAnalyzer(hull_distance=0.1)
            analyzer.plot_hull(df, new_result_ids=["mp-685151", "mp-755875"],
                               filename="hull.png")
            self.assertTrue(os.path.isfile("hull.png"))

        # Test 3D
        with ScratchDir('.'):
            analyzer.hull_distance = 0.05
            analyzer.plot_hull(df, new_result_ids=["mp-776280", "mp-30998"],
                               filename="hull.png")
            self.assertTrue(os.path.isfile("hull.png"))

    def test_structure_analyzer(self):
        jobs = loadfn(os.path.join(CAMD_TEST_FILES, "raw_results.json"))
        analyzer = AnalyzeStructures()
        self.assertEqual(analyzer.analyze_vaspqmpy_jobs(jobs, against_icsd=True, use_energies=False),
                         [True, True, True, False, True, False, True, True, True])

        self.assertEqual(analyzer.analyze_vaspqmpy_jobs(jobs, against_icsd=True, use_energies=True),
                         [True, True, False, True, True, False, True, True, True])

    def test_analyze(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = df['formula']
        analyzer = StabilityAnalyzer(hull_distance=0.1)
        stab_new, stab_all = analyzer.analyze(
            df, new_result_ids=["mp-30998", "mp-572822"],
            all_result_ids=["mp-30998", "mp-572822", "mp-754790", "mp-656850"],
            return_within_hull=False)
        self.assertAlmostEqual(stab_new[0], 0)
        self.assertAlmostEqual(stab_new[1], 0.52784795)
        stab_new, stab_all = analyzer.analyze(
            df, new_result_ids=["mp-30998", "mp-572822"],
            all_result_ids=["mp-30998", "mp-572822", "mp-754790", "mp-656850"],
        )
        self.assertTrue(stab_new[0])
        self.assertFalse(stab_new[1])
        self.assertTrue(stab_all[2])
        self.assertFalse(stab_all[3])

        # Test non-included result_ids, this one is "mp-1235"
        stab_new, stab_all = analyzer.analyze(
            df, new_result_ids=["mp-73", "mp-72", "mp-1057818", "mp-1235"],
            all_result_ids=["mp-73", "mp-72", "mp-46", "mp-1057818"]
        )


if __name__ == '__main__':
    unittest.main()
