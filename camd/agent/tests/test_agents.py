import unittest
import pandas
import numpy as np

from camd.agent.generic import LinearAgent


class GenericAgentsTest(unittest.TestCase):
    def test_linear_agent(self):
        """
        Verify that linear agent can fit a linear function and recover the minimizing argument to new data.
        """

        X = np.linspace(-5, 5, 5)
        Y = 2 * X + 1

        df = pandas.DataFrame(
            data=np.vstack((X, Y)).T, index=range(5), columns=["domain", "target"]
        )

        candidate_X = np.linspace(-10, 10, 10)
        candidate_df = pandas.DataFrame(data=candidate_X, columns=["domain"])

        agent = LinearAgent()

        predictions = agent.get_hypotheses(seed_data=df, candidate_data=candidate_df)
        assert predictions["domain"][9] == 10
