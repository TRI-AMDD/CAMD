# Copyright Toyota Research Institute 2019

import abc

from monty.json import MSONable
from camd.log import camd_traced


# TODO: rethink naming here, should it be ExperimentHandler?
# TODO: rethink history storage
class Experiment(abc.ABC, MSONable):
    """
    An abstract class for brokering experiments, this is a
    temporary placeholder for the way we do experiments

    Eventually, this step should execute asynchronously,
    we might just want to use FireWorks for this
    """

    def __init__(self, current_data=None, job_status=None,
                 history=None):
        self.current_data = current_data
        self.job_status = job_status
        self._history = history or []

    def update_current_data(self, data):
        """
        Updates current data with dataframe,
        stores old data in history

        Args:
            data (DataFrame):

        Returns:
            None

        """
        if self.current_data is not None:
            current_results = self.get_results()
            self._history.append((self.current_data, current_results))

        self.current_data = data

    @abc.abstractmethod
    def get_results(self):
        """
        Args:
            indices (list): uids / indices of experiments to get the results for
        Returns:
            dict: a dictionary of results

        """

    @abc.abstractmethod
    def monitor(self):
        """
        Keeps track of jobs given the poll_time and timeout
        """

    @abc.abstractmethod
    def submit(self, data):
        """
        Args:
            data (DataFrame): dataframe containing all necessary
                data to conduct the experiment(s).  May be one
                row, may be multiple rows

        Returns:
            None
        """


@camd_traced
class ATFSampler(Experiment):
    """
    A simple after the fact sampler that just samples
    a dataframe according to index_values
    """
    def __init__(self, dataframe, current_data=None, job_status=None):
        self.dataframe = dataframe
        super(ATFSampler, self).__init__(
            current_data=current_data,
            job_status=job_status
        )

    def get_results(self):
        """
        Simply samples the dataframe associated with the ATFSampler
        object according to the last submitted data

        Returns:
            (DataFrame): DataFrame of results

        """
        indices = self.current_data.index
        return self.dataframe.loc[indices].dropna(axis=0, how='any')

    def submit(self, data):
        """This does nothing, since the "experiments" are already done"""
        self.current_data = data
        self.job_status = "COMPLETED"
        return None

    def monitor(self):
        return True


