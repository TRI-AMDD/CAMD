import unittest
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from monty.serialization import loadfn
from m3gnet.models import M3GNet

from camd import CAMD_TEST_FILES
from camd.agent.m3gnet import M3GNetAgent


class StabilityAgentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data = pd.read_pickle(os.path.join(CAMD_TEST_FILES, 'test_m3gnet.pickle'))
        cls.test_data = test_data
        cls.seed_data, cls.candidate_data = train_test_split(
            test_data, train_size=0.2, random_state=42
        )
        model = M3GNet(is_intensive=False)
        cls.agent = M3GNetAgent(m3gnet=model)

    def test_reverse_calcs(self):
        X_struct, Xe, Xf, Xs = self.agent.reverse_calcs(self.test_data['calcs_reversed'].iloc[:2])
        self.assertEqual(len(X_struct), 90)
        self.assertTrue(Xe[0] - -3.5534271704166667 <= 1e-4)
        self.assertTrue((np.array(Xf[0]) - np.array([[-0.03578407, -0.00176274, 0.03967705],
 [0.06499199, -0.02277058, 0.03514361],
 [-0.06499199, 0.02277058, -0.03514361],
 [0.03578407, 0.00176274, -0.03967705],
 [-0.03578407, 0.00176274, 0.03967705],
 [0.06499199, 0.02277058, 0.03514361],
 [-0.06499199, -0.02277058, -0.03514361],
 [0.03578407, -0.00176274, -0.03967705],
 [0.0, 0.02825717, 0.0],
 [0.0, 0.02280701, 0.0],
 [-0.02465965, -0.00049394, -0.04887495],
 [-0.02165852, 0.02248651, -0.04979937],
 [-0.01744417, -0.02081133, 0.03051514],
 [0.01744417, 0.02081133, -0.03051514],
 [0.02165852, -0.02248651, 0.04979937],
 [0.02465965, 0.00049394, 0.04887495],
 [0.0, -0.02280701, 0.0],
 [0.0, -0.02825717, 0.0],
 [-0.02465965, 0.00049394, -0.04887495],
 [-0.02165852, -0.02248651, -0.04979937],
 [-0.01744417, 0.02081133, 0.03051514],
 [0.01744417, -0.02081133, -0.03051514],
 [0.02165852, 0.02248651, 0.04979937],
 [0.02465965, -0.00049394, 0.04887495]])).max() <= 1e-4 )
        
    #def test_train(self):
    #    X_struct, Xe, Xf, Xs = self.agent.reverse_calcs(self.test_data['calcs_reversed'].iloc[:2])
    #    self.agent.train(self.seed_data, epochs=1)
        
    def test_hypotheses(self):
        self.agent.get_hypotheses(candidate_data=self.candidate_data, seed_data=self.seed_data, retrain=False)

        
if __name__ == '__main__':
    unittest.main()
