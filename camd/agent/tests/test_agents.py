import unittest
import pandas
import numpy as np

from camd.agent.generic import LinearAgent
from camd.agent.base import SequentialAgent


class GenericAgentsTest(unittest.TestCase):
    def setUp(self):
        X = np.linspace(-5, 5, 5)
        Y = 2 * X + 1

        self.seed_data = pandas.DataFrame(
            data=np.vstack((X, Y)).T, index=range(5), columns=["domain", "target"]
        )

        candidate_X = np.linspace(-10, 10, 10)
        self.candidate_df = pandas.DataFrame(data=candidate_X, columns=["domain"])

    def test_linear_agent(self):
        """
        Verify that linear agent can fit a linear
        function and recover the minimizing argument to new data.
        """
        agent = LinearAgent(n_query=10)
        predictions = agent.get_hypotheses(
            seed_data=self.seed_data, candidate_data=self.candidate_df
        )
        self.assertEqual(predictions["domain"][9], 10)

    def test_sequential_agent(self):
        agents = [LinearAgent(n_query=10), LinearAgent(n_query=5)]
        agent = SequentialAgent(agents=agents)
        agent.get_hypotheses(self.seed_data, self.candidate_df)
        self.assertEqual(len(get_hypotheses), 5)
