import unittest
import os
from camd.utils.data import cache_matrio_data, load_dataframe
from camd import CAMD_CACHE


class DataTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
