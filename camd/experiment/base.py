# Copyright Toyota Research Institute 2019

import abc

from monty.json import MSONable


class Experiment(abc.ABC, MSONable):
    """
    An abstract class for brokering experiments, this is a
    temporary placeholder for the way we do experiments

    Eventually, this step should execute asynchronously,
    we might just want to use FireWorks for this
    """

    def __init__(self, params):
        self._params = params
        self.unique_ids = params['unique_ids'] if 'unique_ids' in params else []
        self.job_status = params['job_status'] if 'job_status' in params else {}


    @abc.abstractmethod
    def get_state(self):
        """
        # TODO: refine this into something more universal
        Returns:
            str: 'unstarted', 'pending', 'completed'

        """

    @abc.abstractmethod
    def get_results(self, indices):
        """
        Args:
            indices (list): uids / indices of experiments to get the results for
        Returns:
            dict: a dictionary of results

        """

    def get_parameter(self, parameter_name):
        """
        Args:
            parameter_name (str): name of parameter to get

        Returns:
            parameter value

        """
        return self._params.get(parameter_name)


    def _update_results(self):
        self.results = self.get_results()

    @abc.abstractmethod
    def monitor(self):
        """
        Keeps track of jobs given the poll_time and timeout
        """

    @abc.abstractmethod
    def submit(self, unique_ids):
        """
        # Accepts job requests by unique id of candidates
        Returns:
            str: 'unstarted', 'pending', 'completed'

        """

    @classmethod
    def from_job_status(cls, params, job_status):
        params["job_status"] = job_status
        params["unique_ids"] = list(job_status.keys())
        return cls(params)


class ATFSampler(Experiment):
    """
    A simple after the fact sampler that just samples
    a dataframe according to index_values
    """

    def start(self):
        """There's no start procedure for this particular experiment"""
        pass

    def get_state(self):
        """This experiment should be complete on construction"""
        return True

    def get_results(self, index_labels):
        dataframe = self.get_parameter('dataframe')
        return dataframe.loc[index_labels].dropna(axis=0, how='any')

    def submit(self, unique_ids, *args):
        self.unique_ids = unique_ids
        return dict(zip(unique_ids, [{'status': 'SUCCEEDED'} for i in range(len(unique_ids))]))

    def monitor(self):
        unique_ids = self.unique_ids
        return dict(zip(unique_ids, [{'status': 'SUCCEEDED'} for i in range(len(unique_ids))]))
