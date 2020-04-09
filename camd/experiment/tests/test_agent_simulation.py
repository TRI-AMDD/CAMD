#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import pandas as pd
from monty.tempfile import ScratchDir
from camd.utils.data import load_dataframe, partition_intercomp, \
    get_oqmd_data_by_chemsys
from camd.agent.base import RandomAgent
from camd.experiment.agent_simulation import LocalAgentSimulation
from camd.analysis import StabilityAnalyzer


class CampaignSimulationTest(unittest.TestCase):
    def test_run(self):
        with ScratchDir('.'):
            dataframe = get_oqmd_data_by_chemsys("Fe-O")
            cand, seed = partition_intercomp(dataframe, n_elements=1)
            agents_df = pd.DataFrame({"agent": [RandomAgent()]})
            simulation = LocalAgentSimulation(
                cand, iterations=5, seed_data=seed,
                analyzer=StabilityAnalyzer())
            simulation.submit(agents_df)
            simulation.monitor()
            results = simulation.get_results()
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
