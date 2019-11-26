#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import os

from pymatgen.util.testing import PymatgenTest
from pymatgen import MPRester
from camd.experiment.dft import OqmdDFTonMC1
import pandas as pd


CAMD_DFT_TESTS = os.environ.get("CAMD_DFT_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_DFT_TESTS to run long tests"


# This test is still inconsistent because of issues with
# batch AWS jobs and database communications
class Mc1Test(unittest.TestCase):
    @unittest.skipUnless(CAMD_DFT_TESTS, SKIP_MSG)
    def test_get(self):
        good_silicon = PymatgenTest.get_structure("Si")
        bad_silicon = good_silicon.copy()

        # Add another site at the same position
        bad_silicon.append("Si", [0.1, 0.1, 0.15])
        bad_silicon.append("Si", [0.1, 0.333, 0.15])
        self.assertEqual(len(bad_silicon), 4)
        data = pd.DataFrame(
            {"structure": {
                "good": good_silicon,
                "bad": bad_silicon
            }
            }
        )

        experiment = OqmdDFTonMC1(poll_time=30, timeout=150)
        experiment.submit(data)
        status = experiment.monitor()
        results = experiment.get_results()

        self.assertAlmostEqual(results.loc['good', 'delta_e'], 0, 5)
        self.assertIsNone(results.loc['bad', 'result'])

    @unittest.skipUnless(CAMD_DFT_TESTS, SKIP_MSG)
    def test_structure_suite(self):
        mp_ids = ["mp-702",
                  "mp-1953",
                  "mp-1132",
                  "mp-8409",
                  "mp-872"]
        import nose; nose.tools.set_trace()
        with MPRester() as mpr:
            structure_dict = {mp_id: mpr.get_structure_by_material_id(mp_id)
                              for mp_id in mp_ids}
        data = pd.DataFrame({"structure": structure_dict})

        experiment = OqmdDFTonMC1(poll_time=25)
        experiment.submit(data)
        status = experiment.monitor()
        results = experiment.get_results()
        self.assertTrue(all(results['status'] == "SUCCEEDED"))


if __name__ == '__main__':
    unittest.main()
