#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import os
import pandas as pd
from pymatgen import Composition
from camd import CAMD_TEST_FILES
from camd.analysis import AnalyzeStability_mod


class AnalysisTest(unittest.TestCase):
    def test_present(self):
        df = pd.read_csv(os.path.join(CAMD_TEST_FILES, "test_df_analysis.csv"),
                         index_col="id")
        df['Composition'] = [Composition(f) for f in df['formula']]
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


if __name__ == '__main__':
    unittest.main()
