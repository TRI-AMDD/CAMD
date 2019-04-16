# Copyright Toyota Research Institute 2019

import abc
from monty.serialization import MSONable

#TODO: Experiment Broker


def get_dft_calcs_aft(uids, df):
    """
    Mock function that mimics fetching DFT calculations
    """
    uids = [uids] if type(uids) != list else uids
    return df.loc[uids]


def get_dft_calcs_from_northwestern(uids):
    """
    Placeholder function for fetching DFT calculations from Northwestern
    """
    pass


def get_dft_calcs_from_MC1(uids):
    """
    Placeholder function for fetching DFT calculations from MC1
    """
    pass


class Experiment(abc.ABC, MSONable):
    """
    An abstract class for brokering experiments, this is a
    temporary placeholder for the way we do experiments

    Eventually, this step should execute asynchronously
    """

    def __init__(self):
        self._state = 'unstarted'

    @abc.abstractmethod
    def start(self, params):
        """

        Args:
            params:

        Returns:

        """
        self._state = "pending"

    @property
    @abc.abstractmethod
    def state(self):
        """
        # TODO: refine this into something more universal
        Returns:
            str: 'unstarted', 'pending', 'completed'

        """
        return self._state

    @abc.abstractmethod
    def refresh(self):
        """
        Returns:
        """

    @abc.abstractmethod
    def get_results(self):
        """

        Returns:
            dict: a dictionary of results

        """

    def wait_until_complete(self, time_interval=120):
        """

        Args:
            time_interval:

        Returns:

        """
        state = self.state
        while state is not 'completed':
            self.sleep(time_interval)
            state = self.state
        return state

    def run_and_get_results(self, params):
        """

        Args:
            params:

        Returns:

        """
        self.start_experiment(params)
        self.wait_until_complete()
        return self.get_results()
