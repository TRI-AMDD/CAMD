#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import os
import pandas as pd
import json
from pymatgen import Composition
from monty.serialization import loadfn
from camd import CAMD_TEST_FILES
from camd.analysis import AnalyzeStability_mod, AnalyzeStructures


class AnalysisTest(unittest.TestCase):
    def test_present(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = df['formula']
        # Test 2D
        analyzer = AnalyzeStability_mod(df, hull_distance=0.1)
        analyzer.present(
            df,
            all_result_ids=["mp-8057", "mp-882", "mp-753593", "mvc-4715"],
            new_result_ids=["mp-685151", "mp-755875"])

        # Test 3D
        analyzer.hull_distance = 0.05

        analyzer.present(
            df,
            all_result_ids=["mp-754790", "mvc-4715"],
            new_result_ids=["mp-776280", "mp-30998"]
        )

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
        analyzer = AnalyzeStability(df, hull_distance=0.1)
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
