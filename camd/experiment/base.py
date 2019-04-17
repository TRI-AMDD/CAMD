# Copyright Toyota Research Institute 2019

import abc

from time import sleep
from monty.json import MSONable


class Experiment(abc.ABC, MSONable):
    """
    An abstract class for brokering experiments, this is a
    temporary placeholder for the way we do experiments

    Eventually, this step should execute asynchronously,
    we might just want to use FireWorks for this
    """

    def __init__(self, params):
        self.state = 'unstarted'
        self._params = params

    @abc.abstractmethod
    def start(self):
        """

        Args:
            params:

        Returns:

        """
        self.state = "pending"

    @abc.abstractmethod
    def get_state(self):
        """
        # TODO: refine this into something more universal
        Returns:
            str: 'unstarted', 'pending', 'completed'

        """

    def _update_state(self):
        """
        Returns:
        """
        self.state = self.get_state()

    @abc.abstractmethod
    def get_results(self):
        """

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

    def _wait_until_complete(self, time_interval=120):
        """

        Args:
            time_interval (float): time interval between
                polling steps

        Returns:
            str: the ending state

        """
        state = self.state
        while self.state is not 'completed':
            self._update_state()
            sleep(time_interval)
        return state

    def run(self):
        """

        Args:
            params:

        Returns:

        """
        self.start()
        self._wait_until_complete()
        self._update_results()
        return True


class ATFSampler(Experiment):
    """
    A simple after the fact sampler that just samples
    a dataframe according to index_values
    """
    def __init__(self, params):
        """

        Args:
            params (dict):
        """
        super(ATFSampler, self).__init__(params)

    def start(self):
        """There's no start procedure for this particular experiment"""
        pass

    def get_state(self):
        """This experiment should be complete on construction"""
        return 'completed'

    def get_results(self):
        indices = self.get_parameter('index_values')
        dataframe = self.get_parameter('dataframe')
        return dataframe.iloc[indices]
