import unittest

from monty.tempfile import ScratchDir
from pymatgen.util.testing import PymatgenTest
from camd.experiment import submit_dft_calcs_to_mc1, check_dft_calcs



@unittest.skipIf(True, "toggle this test")
class Mc1Test(unittest.TestCase):
    def test_get(self):
        good_silicon = PymatgenTest.get_structure("Si")
        bad_silicon = good_silicon.copy()

        # Add another site at the same position
        bad_silicon.append("Si", bad_silicon[0].frac_coords)

        with ScratchDir('.'):
            structure_dict = {"good": good_silicon,
                              "bad": bad_silicon}
            calc_status = submit_dft_calcs_to_mc1(structure_dict)
            finished = False
            while not finished:
                calc_status = check_dft_calcs(calc_status)
                print("Calc status: {}".format(calc_status))
                finished = all([doc['status'] is not ['pending']
                                for doc in calc_status.values()])
        self.assertEqual(calc_status['good']['status'], 'completed')
        self.assertEqual(calc_status['bad']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
