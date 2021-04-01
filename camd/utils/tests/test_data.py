import unittest
import os
from camd.utils.data import cache_matrio_data, load_dataframe, \
    partition_intercomp, get_chemsys
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from camd import CAMD_CACHE


class DataCacheTest(unittest.TestCase):
    def setUp(self):
        # Remove smallest file and cache
        self.smallfile = "oqmd1.2_exp_based_entries_featurized_v2.pickle"
        self.smallfile_path = os.path.join(CAMD_CACHE, self.smallfile)
        if os.path.isfile(self.smallfile_path):
            os.remove(self.smallfile_path)

    def test_cache(self):
        cache_matrio_data(self.smallfile)
        self.assertTrue(os.path.isfile(self.smallfile_path))

        # Ensure no re-download of existing files in cache
        time_before = os.path.getmtime(self.smallfile_path)
        cache_matrio_data(self.smallfile)

        time_after = os.path.getmtime(self.smallfile_path)
        self.assertEqual(time_after, time_before)

    def test_load_dataframe(self):
        df_name = self.smallfile.rsplit('.', 1)[0]
        dataframe = load_dataframe(df_name)
        self.assertEqual(len(dataframe), 36581)


class PartitionTest(unittest.TestCase):
    def setUp(self):
        # Remove smallest file and cache
        self.smallfile = "oqmd1.2_exp_based_entries_featurized_v2.pickle"
        self.smallfile_path = os.path.join(CAMD_CACHE, self.smallfile)

    def test_partition(self):
        df_name = self.smallfile.rsplit('.', 1)[0]
        dataframe = load_dataframe(df_name)
        cand, seed = partition_intercomp(dataframe)
        self.assertEqual(len(dataframe), len(cand) + len(seed))

        cand, seed = partition_intercomp(dataframe, n_elements=1)
        self.assertEqual(len(dataframe), len(cand) + len(seed))
        self.assertGreater(len(seed), 0)


class GeneralUtilsTest(unittest.TestCase):
    def test_get_chemsys(self):
        # Test with structure
        struct = Structure.from_spacegroup('Fm-3m', Lattice.cubic(5.0), ['Ni', 'O'],
                                           [[0, 0, 0], [0.75, 0.25, 0.75]])
        self.assertEqual(get_chemsys(struct), 'Ni-O')

        # Test sorting
        test_val = get_chemsys("TiO2")
        self.assertEqual(test_val, "O-Ti")

        test_val = get_chemsys("BaNiO3")
        self.assertEqual(test_val, "Ba-Ni-O")


if __name__ == '__main__':
    unittest.main()
