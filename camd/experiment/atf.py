#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from camd.experiment.base import Experiment, ATFSampler
from camd.loop import Loop


class LocalAgentSimulation(Experiment):
    def __init__(self, atf_dataframe, iterations, analyzer, n_seed,
                 current_data=None, job_status="unstarted"):
        """
        Args:
            atf_dataframe:
            iterations:
            analyzer:
            n_seed:
            current_data:
            job_status:
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
        self.current_data = data
        agent = data['agent']
        loop = Loop(
            candidate_data=self.atf_dataframe,
            agent=agent,
            analyzer=self.analyzer,
            experiment=ATFSampler(dataframe=self.atf_dataframe),
            create_seed=self.n_seed,
        )
        loop.auto_loop(n_iterations=self.iterations, initialize=True)
        self.job_status = "completed"

    def monitor(self):
        pass
