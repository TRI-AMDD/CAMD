#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest

from pymatgen.util.testing import PymatgenTest
from pymatgen import MPRester
from camd.experiment.dft import OqmdDFTonMC1


# This test is still inconsistent because of issues with
# batch AWS jobs and database communications
class Mc1Test(unittest.TestCase):
    @unittest.skipUnless(False, "toggle this test")
    def test_get(self):

        good_silicon = PymatgenTest.get_structure("Si")
        bad_silicon = good_silicon.copy()

        # Add another site at the same position
        bad_silicon.append("Si", [0.1, 0.1, 0.15])
        bad_silicon.append("Si", [0.1, 0.333, 0.15])
        self.assertEqual(len(bad_silicon), 4)
        params = {"structure_dict": {"good": good_silicon,
                                     "bad": bad_silicon},
                  "poll_time": 30, "timeout": 150}

        experiment = OqmdDFTonMC1(params)
        experiment.submit()
        status = experiment.monitor()
        results = experiment.get_results()

        self.assertEqual(results['good']['status'], 'SUCCEEDED')
        self.assertEqual(results['bad']['status'], 'FAILED')

    @unittest.skipUnless(False, "toggle this test")
    def test_structure_suite(self):
        # TODO: fix the formation energy calculation
        mp_ids = ["mp-702",
                  "mp-1953",
                  "mp-1132",
                  "mp-8409",
                  "mp-872"]
        with MPRester() as mpr:
            structure_dict = {mp_id: mpr.get_structure_by_material_id(mp_id)
                              for mp_id in mp_ids}
        params = {"structure_dict": structure_dict,
                  "poll_time": 25}

        experiment = OqmdDFTonMC1(params)
        experiment.submit()
        status = experiment.monitor()
        results = experiment.get_results()
        self.assertTrue(all([run['status'] == "SUCCEEDED" for run in status]))


if __name__ == '__main__':
    unittest.main()