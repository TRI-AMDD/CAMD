#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from camd.experiment.base import Experiment, ATFSampler
from camd.campaigns.base import Campaign
from monty.os import cd
import os


class LocalAgentSimulation(Experiment):
    """
    Class that runs Agent simulations synchronously and
    sequentially for testing in meta-agent campaigns.
    """
    def __init__(self, atf_candidate_data, seed_data, analyzer, iterations,
                 current_data=None, job_status=None):
        """
        Args:
            atf_candidate_data (DataFrame): dataframe corresponding to after
                the fact data to sample
            seed_data (DataFrame): seed data to use for the campaign
            analyzer (Analyzer): Analyzer to use in the loop
            iterations (int): number of iterations to execute
                in the loop
            current_data (dataframe): current data (for restarting)
            job_status (str): job status (for restarting)

        """
        self.atf_dataframe = atf_candidate_data
        self.iterations = iterations
        self.analyzer = analyzer
        self.seed_data = seed_data
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
        if self.job_status == "PENDING":
            campaigns = []
            for index, row in self.current_data.iterrows():
                agent = row.pop('agent')
                path = str(index)
                os.mkdir(path)
                with cd(path):
                    campaigns.append(self.test_agent(agent))
            self.current_data['campaign'] = campaigns
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
        campaign = Campaign(
                candidate_data=self.atf_dataframe,
                seed_data=self.seed_data,
                agent=agent,
                analyzer=self.analyzer,
                experiment=ATFSampler(
                    dataframe=self.atf_dataframe
                ),
            )
        campaign.auto_loop(n_iterations=self.iterations, initialize=True)
        return campaign

    def get_results(self):
        """
        Gets current data corresponding to last run campaign

        Returns:
            (pandas.DataFrame) current data attribute

        """
        return self.current_data
