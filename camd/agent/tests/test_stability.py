import unittest
from sklearn.model_selection import train_test_split
from camd.agent.stability import (
    QBCStabilityAgent,
    AgentStabilityML5,
    GaussianProcessStabilityAgent,
    BaggedGaussianProcessStabilityAgent,
    AgentStabilityAdaBoost,
)
from camd.utils.data import load_default_atf_data


class StabilityAgentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data = load_default_atf_data().sample(200)
        cls.seed_data, cls.candidate_data = train_test_split(
            test_data, train_size=0.2, random_state=42
        )

    def test_qbc_stability_agent(self):
        agent = QBCStabilityAgent()
        hypotheses = agent.get_hypotheses(
            candidate_data=self.candidate_data, seed_data=self.seed_data
        )

    def test_ml_agent(self):
        agent = AgentStabilityML5()
        hypotheses = agent.get_hypotheses(
            candidate_data=self.candidate_data, seed_data=self.seed_data
        )

    def test_gp_stability_agent(self):
        agent = GaussianProcessStabilityAgent()
        hypotheses = agent.get_hypotheses(
            candidate_data=self.candidate_data, seed_data=self.seed_data
        )

    def test_bagged_gp_stability_agent(self):
        agent = BaggedGaussianProcessStabilityAgent(max_samples=30)
        hypotheses = agent.get_hypotheses(
            candidate_data=self.candidate_data, seed_data=self.seed_data
        )

    def test_agent_stability_adaboost(self):
        agent = AgentStabilityAdaBoost()
        hypotheses = agent.get_hypotheses(
            candidate_data=self.candidate_data, seed_data=self.seed_data
        )


if __name__ == "__main__":
    unittest.main()
