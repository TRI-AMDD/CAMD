import unittest
import time

from monty.tempfile import ScratchDir
from pymatgen.util.testing import PymatgenTest
from camd.experiment import submit_dft_calcs_to_mc1, check_dft_calcs,\
    run_dft_experiments


@unittest.skipIf(False, "toggle this test")
class Mc1Test(unittest.TestCase):
    def test_get(self):
        good_silicon = PymatgenTest.get_structure("Si")
        bad_silicon = good_silicon.copy()

        # Add another site at the same position
        bad_silicon.append("Si", [0.1, 0.1, 0.15])
        bad_silicon.append("Si", [0.1, 0.333, 0.15])
        self.assertEqual(len(bad_silicon), 4)
        calc_status = run_dft_experiments({
            "good": good_silicon, "bad": bad_silicon},
             poll_time=30, timeout=200)
        # with ScratchDir('.'):
        #     structure_dict =
        #     calc_status = submit_dft_calcs_to_mc1(structure_dict)
        #     finished = False
        #     while not finished:
        #         time.sleep(30)
        #         calc_status = check_dft_calcs(calc_status)
        #         print("Calc status: {}".format(calc_status))
        #         finished = all([doc['status'] in ['SUCCEEDED', 'FAILED']
        #                         for doc in calc_status.values()])

        self.assertEqual(calc_status['good']['status'], 'COMPLETED')
        self.assertEqual(calc_status['bad']['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
