import unittest
import os
import numpy as np

from pymatgen.core import Structure
from pymatgen.util.testing import PymatgenTest
from camd.experiment.dft import AtomateExperiment, get_mp_formation_energy
import pandas as pd
from fireworks import LaunchPad

TEST_DIR = os.path.dirname(__file__)
ATOMATE_DFT_TESTS = np.all(
    [
        os.path.isfile(os.path.join(TEST_DIR, i))
        for i in ["db.json", "my_launchpad.yaml"]
    ]
)
SKIP_MSG = (
    "Long tests disabled, provide db.json, my_launchpad.yaml "
    "in {} to run long tests".format(TEST_DIR)
)


class AtomateTest(PymatgenTest):
    # @unittest.skipUnless(ATOMATE_DFT_TESTS, SKIP_MSG)
    # def test_get(self) -> None:
    #     good_silicon = PymatgenTest.get_structure("Si")
    #     bad_silicon = good_silicon.copy()
    #
    #     # Add another site at the same position
    #     bad_silicon.append("Si", [0.1, 0.1, 0.15])
    #     bad_silicon.append("Si", [0.1, 0.333, 0.15])
    #     self.assertEqual(len(bad_silicon), 4)
    #     data = pd.DataFrame({"structure": {"good": good_silicon, "bad": bad_silicon}})
    #     db_file = os.path.join(TEST_DIR, "db.json")
    #     lpad_file = os.path.join(TEST_DIR, "my_launchpad.yaml")
    #     lpad = LaunchPad.from_file(lpad_file)
    #     lpad.auto_load()
    #     experiment = AtomateExperiment(
    #         lpad, db_file, poll_time=30, launch_from_local=False
    #     )
    #     # experiment.submit(data)
    #     # status = experiment.monitor()
    #     results = experiment.get_results()
    #
    #     self.assertAlmostEqual(
    #         results.loc["good", "final_energy_per_atom"], -5.420645, 5
    #     )
    #     self.assertIsNone(results.loc["bad", "task_id"])
    #     self.assertEqual(experiment.job_status, "COMPLETED")

    @unittest.skipUnless(ATOMATE_DFT_TESTS, SKIP_MSG)
    def test_delta_e(self) -> None:
        fe2o3 = Structure.from_file(os.path.join(TEST_DIR, "fe2o3.cif"))  # mp-19770
        data = pd.DataFrame(
            {"structure": {"mp-19770": fe2o3}, "wf_name": "opt_mp-19770", "fw_id": 2749}
        )
        db_file = os.path.join(TEST_DIR, "db.json")
        lpad_file = os.path.join(TEST_DIR, "my_launchpad.yaml")
        lpad = LaunchPad.from_file(lpad_file)
        lpad.auto_load()
        experiment = AtomateExperiment(
            lpad, db_file, poll_time=30, launch_from_local=False
        )
        experiment.update_current_data(data)
        experiment.update_results()
        self.assertAlmostEqual(experiment.current_data["delta_e"].values[0], -1.9072, 5)

    @unittest.skipUnless(ATOMATE_DFT_TESTS, SKIP_MSG)
    def testCache(self):
        db_file = os.path.join(TEST_DIR, "db.json")
        lpad_file = os.path.join(TEST_DIR, "my_launchpad.yaml")
        lpad = LaunchPad.from_file(lpad_file)
        lpad.auto_load()
        current_data = pd.read_pickle(os.path.join(TEST_DIR, "Si_test.p"))
        experiment = AtomateExperiment(
            lpad, db_file, poll_time=30, launch_from_local=False
        )
        experiment.update_current_data(current_data)
        experiment.update_results()
        results = experiment.get_results()
        self.assertAlmostEqual(
            results.loc["good", "final_energy_per_atom"], -5.420645, 5
        )
        self.assertIsNone(results.loc["bad", "task_id"])
        self.assertEqual(experiment.job_status, "COMPLETED")

    def test_eform(self):
        # SiO2 mp-546794, Fe2O3 mp-19770
        # mp-19770
        # uncorrected e -67.4928
        # corrected e -80.6388
        total_e = -67.491217
        formula = "Fe4O6"
        potcar_symbols = ["PBE Fe_pv", "PBE O"]
        hubbards = {"Fe": 5.3, "O": 0.0}
        self.assertAlmostEqual(
            get_mp_formation_energy(total_e, formula, potcar_symbols, hubbards),
            -1.9072,
            5,
        )


if __name__ == "__main__":
    unittest.main()
