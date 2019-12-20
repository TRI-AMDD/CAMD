#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
from monty.tempfile import ScratchDir
from camd.utils.data import load_default_atf_data
from camd.agent.base import RandomAgent
from camd.experiment.agent_simulation import LocalAgentSimulation
from camd.analysis import StabilityAnalyzer


class CampaignSimulationTest(unittest.TestCase):
    def test_run(self):
        with ScratchDir('.'):
            df = load_default_atf_data()
            agents_df = pd.DataFrame({"agent": [RandomAgent()]})
            simulation = LocalAgentSimulation(
                df, iterations=5, n_seed=10, analyzer=StabilityAnalyzer())
            simulation.submit(agents_df)
            simulation.monitor()
            results = simulation.get_results()
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
