#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from camd.experiment.base import Experiment, ATFSampler
from camd.loop import Loop


class LocalAgentSimulation(Experiment):
    def __init__(self, atf_dataframe, iterations, analyzer, n_seed,
                 current_data=None, job_status=None):
        """
        Args:
            atf_dataframe (DataFrame):
            iterations (int): number of iterations to execute
                in the loop
            analyzer (Analyzer): Analyzer to use in the loop
            n_seed (int): number of points to use in the seed data
            current_data (dataframe): current data (for restarting)
            job_status (str): job status (for restarting)
        """
        self.atf_dataframe = atf_dataframe
        self.iterations = iterations
        self.analyzer = analyzer
        self.n_seed = n_seed
        super(LocalAgentSimulation, self).__init__(
            current_data=current_data, job_status=job_status)

    def submit(self, data):
        """
        Args:
            data (DataFrame): data associated with agent

        Returns:
            None

        """
        self.update_current_data(data)
        self.job_status = 'PENDING'

    def monitor(self):
        """
        The monitor method in the case just runs the necessary
        agent simulation for all specified agents

        Returns:
            None

        """
        agents = self.current_data['agent']
        for agent in agents:
            loop = self.test_agent(agent)
        self.job_status = "COMPLETED"

    def test_agent(self, agent):
        """
        Runs a simulation of a given agent according to the
        class attributes

        Args:
            agent (HypothesisAgent):

        Returns:
            None

        """
        loop = Loop(
                candidate_data=self.atf_dataframe,
                agent=agent,
                analyzer=self.analyzer,
                experiment=ATFSampler(
                    dataframe=self.atf_dataframe
                ),
                create_seed=self.n_seed,
            )
        loop.auto_loop(n_iterations=self.iterations, initialize=True)
        return loop

    def get_results(self):
        return self.current_data
