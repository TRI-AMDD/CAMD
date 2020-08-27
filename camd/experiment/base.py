# Copyright Toyota Research Institute 2019
"""
Contains basic logic and abstraction for experiments and
generic experiments like the ATF Sampler.
"""

import abc

import pandas as pd
from monty.json import MSONable


# TODO: rethink naming here, should it be ExperimentHandler?
# TODO: rethink history storage
class Experiment(abc.ABC, MSONable):
    """
    An abstract class for brokering experiments, this is a
    temporary placeholder for the way we do experiments

    Eventually, this step should execute asynchronously,
    we might just want to use FireWorks for this
    """

    def __init__(self, current_data=None, job_status=None, history=None):
        """
        Initializes an experiment.

        Args:
            current_data (pandas.DataFrame): dataframe corresponding to
                currently submitted experiments
            job_status (str): status of the experiment
            history (pandas.DataFrame): history of past experiments
        """
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

    @property
    def agg_history(self):
        """
        Aggregated history, i.e. in two single dataframes
        corresponding to "current data" attributes and
        results

        Returns:
            (DataFrame): history of current data
            (DataFrame): history of results

        """
        cd_list, cr_list = zip(*self._history)
        return pd.concat(cd_list), pd.concat(cr_list)


class ATFSampler(Experiment):
    """
    A simple after the fact sampler that just samples
    a dataframe according to index_values
    """

    def __init__(self, dataframe, current_data=None, job_status=None):
        """
        Initializes an ATFSampler Experiment

        Args:
            dataframe (pandas.DataFrame): dataframe of a-priori results
                from which to pull
            current_data (pandas.DataFrame): data of currently submitted
                "experiments"
            job_status (str): status of job

        """
        self.dataframe = dataframe
        super(ATFSampler, self).__init__(
            current_data=current_data, job_status=job_status
        )

    def get_results(self):
        """
        Simply samples the dataframe associated with the ATFSampler
        object according to the last submitted data

        Returns:
            (DataFrame): DataFrame of results

        """
        indices = self.current_data.index
        return self.dataframe.loc[indices]

    def submit(self, data):
        """
        This does nothing other than update current_data,
        since the "experiments" are already done
        """
        self.update_current_data(data)
        self.job_status = "COMPLETED"
        return None

    def monitor(self):
        """
        ATF Sampler returns results synchronously,
        and thus needs no monitor, so this simply
        returns True

        Returns:
            True

        """
        return True
