#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from camd.experiment.base import Experiment, ATFSampler
from camd.analysis import AnalyzeStability
from camd.loop import Loop
from camd.utils.data import load_default_atf_data


class LocalAgentSimulation(Experiment):
    def __init__(self, params):
        self.dataframe = params.get("dataframe", load_default_atf_data())
        self.iterations = params.get("iterations", 50)
        self.analyzer = params.get("analyzer", AnalyzeStability(hull_distance=0.05))
        self.n_seed = params.get("n_seed", 500)
        self.job_status = "unstarted"
        super(LocalAgentSimulation, self).__init__(params)

    def submit(self, agent):
        loop = Loop(
            candidate_data=self.params['dataframe'],
            agent=self.params['agent'],
            analyzer=self.params['analyzer'],
            experiment=ATFSampler({"dataframe": self.dataframe}),
            create_seed=self.n_seed,
        )
        loop.auto_loop(n_iterations=self.iterations, initialize=True)
        self.update_job_status("completed")

    def monitor(self):
        pass
