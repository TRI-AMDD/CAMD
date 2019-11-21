import unittest

import pandas as pd
import numpy as np
from camd.experiment.base import ATFSampler


class ATFSamplerTest(unittest.TestCase):
    # def setUp(self):
    #     test_dataframe = pd.DataFrame({"index": np.arange(5),
    #                                    "squared": np.arange(5) ** 2})
    #     # params = {"dataframe": test_dataframe,
    #     #           "index_values": [0, 2, 3]}
    #     self.simple_exp = ATFSampler(dataframe=test_dataframe)

    def test_submit_get_results(self):
        test_dataframe = pd.DataFrame({"index": np.arange(5),
                                       "squared": np.arange(5) ** 2})
        simple_exp = ATFSampler(test_dataframe)
        self.assertEqual(simple_exp.get_state(), "completed")
        simple_exp.submit(test_dataframe.loc[[0, 2, 3]])
        simple_exp.monitor()
        self.assertTrue(
            (simple_exp.get_results()['squared'] == [0, 4, 9]).all())


if __name__ == '__main__':
    unittest.main()
