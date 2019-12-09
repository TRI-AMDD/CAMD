# Copyright Toyota Research Institute 2019
import os
import pickle
import json
import time
import numpy as np
import pandas as pd
import shutil

from monty.json import MSONable
from camd.utils.s3 import cache_s3_objs, s3_sync
from camd import S3_CACHE, CAMD_S3_BUCKET
from camd.agent.base import RandomAgent
from camd.log import camd_traced
from pymatgen.util.plotting import pretty_plot


@camd_traced
class Loop(MSONable):
    def __init__(self, candidate_data, agent, experiment, analyzer,
                 finalizer=None, seed_data=None, create_seed=False,
                 heuristic_stopper=np.inf, s3_prefix=None,
                 s3_bucket=CAMD_S3_BUCKET):
        """
        Loop provides a sequential, workflow-like capability where an
        Agent iterates over a candidate space to choose and execute
        new Experiments, given a certain objective. The abstraction
        follows closely the "scientific method". Agent is the entity
        that suggests new Experiments.

        Supporting entities are Analyzers and Finalizers. Framework
        is flexible enough to implement many sequential learning or
        optimization tasks, including active-learning, bayesian optimization
        or black-box optimization with local or global optima search.

        Args:
            candidate_data (pd.DataFrame): List of uids for candidate
                search space for active learning
            agent (HypothesisAgent): a subclass of HypothesisAgent
            experiment (Experiment): a subclass of Experiment
            analyzer (Analyzer): a subclass of Analyzer
            finalizer (obj): needs to have a finalize method.
            seed_data (pandas.DataFrame): Seed Data for active learning,
                index is to be the assumed uid
            create_seed (int): an initial seed size to create from the data
            heuristic_stopper (int): If int, the heuristic stopper will
                kick in to check if loop should be terminated after this
                many iterations, if no discoveries in past #n loops.
            s3_prefix (str): prefix which to prepend all s3 synced files with,
                if None is specified, s3 syncing will not occur
            s3_bucket (str): bucket name for s3 sync.  If not specified,
                CAMD will sync to the specified environment variable.
        """
        # Cloud parameters
        self.s3_prefix = s3_prefix
        self.s3_bucket = s3_bucket

        # Data parameters
        self.candidate_data = candidate_data
        self.seed_data = seed_data if seed_data is not None else pd.DataFrame()
        self.create_seed = create_seed
        self.history = pd.DataFrame()

        # Object parameters
        self.agent = agent
        self.experiment = experiment
        self.analyzer = analyzer
        self.finalizer = finalizer

        # Other parameters
        # TODO: think about how to abstract this away from the loop
        self.heuristic_stopper = heuristic_stopper

        # Internal data
        self._exp_raw_results = None

        # Check if there exists earlier iterations
        if os.path.exists(os.path.join(self.path, 'iteration.json')):
            self.load('iteration')
            self.initialized = True
        else:
            self.iteration = 0
            self.initialized = False

        if self.initialized:
            self.create_seed = False
            self.load('job_status')
            self.experiment.job_status = self.job_status
            self.load('submitted_experiment_requests')
            self.load('seed_data', method='pickle')
            self.load('consumed_candidates')
            self.load('loop_state', no_exist_fail=False)
            self.initialized = True
        else:
            self.submitted_experiment_requests = []
            self.consumed_candidates = []
            self.job_status = {}
            self.initialized = False
            self.loop_state = "UNSTARTED"

    def run(self, finalize=False):
        """
        This method applies a single iteration of the loop, and
        keeps record of everything in place.

        Each iteration consists of:
            1. Get results of requested experiments
            2. Load, Expand, Save seed_data
            3. Augment candidate_space
            4. Analyze results - Stop / Go
            5. Hypothesize
            6. Submit new experiments
        """
        if not self.initialized:
            raise ValueError("Loop must be initialized.")

        # Get new results
        print("Loop {} state: Getting new results".format(self.iteration))
        self.load('submitted_experiment_requests')
        new_experimental_results = self.experiment.get_results()
        os.chdir(self.path)

        # Load seed_data
        self.load('seed_data', method='pickle')

        # Analyze new results
        print("Loop {} state: Analyzing results".format(self.iteration))
        summary, new_seed_data = self.analyzer.analyze(
            self.seed_data, new_experimental_results
        )

        # Augment summary and seed
        self.history.append(summary)
        self.save("history", method="pickle")
        self.seed_data = new_seed_data
        self.save('seed_data', method='pickle')

        # Remove candidates from candidate space
        candidate_space = self.candidate_data.index.difference(
            new_experimental_results.index, sort=False).tolist()
        self.candidate_data = self.candidate_data.loc[candidate_space]

        # Loop stopper if no discoveries in last few cycles.
        if self.iteration > self.heuristic_stopper:
            new_discoveries = self.history['N_Discovery'][-3:].values.sum()
            if new_discoveries == 0:
                self.finalize()
                print("Not enough new discoveries. Stopping the loop.")
                return True

        # Loop stopper if finalization is desired but will be done
        # outside of run (e.g. auto_loop)
        if finalize:
            return None

        # Agent suggests new experiments
        print("Loop {} state: Agent {} hypothesizing".format(
            self.iteration, self.agent.__class__.__name__))
        suggested_experiments = self.agent.get_hypotheses(
            self.candidate_data, self.seed_data)

        # Loop stopper if agent doesn't have anything to suggest.
        if len(suggested_experiments) == 0:
            self.finalize()
            print("No space left to explore. Stopping the loop.")
            return True

        # Experiments submitted
        print("Loop {} state: Running experiments".format(self.iteration))
        experiment_data = self.candidate_data.loc[suggested_experiments]
        self.job_status = self.experiment.submit(experiment_data)
        self.save("job_status")

        self.submitted_experiment_requests = suggested_experiments
        self.save('submitted_experiment_requests')

        self.consumed_candidates += suggested_experiments
        self.save('consumed_candidates')

        self.iteration += 1
        self.save("iteration")

    def auto_loop(self, n_iterations=10, timeout=10, monitor=False,
                  initialize=False, with_icsd=False):
        """
        Runs the loop repeatedly, and locally. Pretty light weight,
        but recommended method is auto_loop_in_directories.
        TODO: Stopping criterion from Analyzer

        Args:
            n_iterations (int): Number of iterations.
            timeout (int): Time (in seconds) to wait on idle for
                submitted experiments to finish.
            monitor (bool): Use Experiment's monitor method to
                keep track of requested experiments.
            initialize (bool): whether to initialize the loop
                before starting
            with_icsd (bool): whether to initialize from icsd

        """
        if initialize:
            if with_icsd:
                self.initialize_with_icsd_seed()
            else:
                self.initialize()
            time.sleep(timeout)
        while n_iterations - self.iteration >= 0:
            print("Iteration: {}".format(self.iteration))
            self.run()
            print("  Waiting for next round ...")
            if monitor:
                self.experiment.monitor()
            time.sleep(timeout)
        self.run(finalize=True)
        self.finalize()

    def auto_loop_in_directories(self, n_iterations=10, timeout=10,
                                 monitor=False, initialize=False,
                                 with_icsd=False):
        """
        Runs the loop repeatedly
        TODO: Stopping criterion from Analyzer
        Args:
            n_iterations (int): Number of iterations.
            timeout (int): Time (in seconds) to wait on idle for
                submitted experiments to finish.
            monitor (bool): Use Experiment's monitor method to keep
                track of requested experiments. Note, if this is set
                True, timeout also needs to be adjusted. If this is
                not set True, make sure timeout is sufficiently long.
            initialize (bool): whether to initialize the loop
            with_icsd (bool): whether to initialize with icsd seed

        """
        if initialize:
            self.loop_state = 'AGENT'
            self.save("loop_state")

            if with_icsd:
                self.initialize_with_icsd_seed()
            else:
                self.initialize()

            self.loop_state = 'EXPERIMENTS STARTED'
            self.save("loop_state")

            if monitor:
                self.experiment.monitor()
                os.chdir(self.path)
            time.sleep(timeout)

            if self.experiment.get_state():
                self._exp_raw_results = self.experiment.job_status
                self.save('_exp_raw_results')

            loop_backup(self.path, '-1')
            self.loop_state = 'EXPERIMENTS COMPLETED'
            self.save("loop_state")

        while n_iterations - self.iteration >= 0:
            self.load("loop_state")

            if self.loop_state in ["EXPERIMENTS COMPLETED", "AGENT"]:
                print("Iteration: {}".format(self.iteration))
                self.loop_state = 'AGENT'
                self.save("loop_state")
                self.run()
                print("  Waiting for next round ...")

            self.loop_state = 'EXPERIMENTS STARTED'
            self.save("loop_state")
            if monitor:
                self.experiment.monitor()
                os.chdir(self.path)
            time.sleep(timeout)
            if self.experiment.get_state():
                self._exp_raw_results = self.experiment.job_status
                self.save('_exp_raw_results')

            loop_backup(self.path, str(self.iteration - 1))
            self.loop_state = 'EXPERIMENTS COMPLETED'
            self.save("loop_state")
        self.run(finalize=True)
        self.finalize()

    def initialize(self, random_state=42):
        if self.initialized:
            raise ValueError(
                "Initialization may overwrite existing loop data. Exit.")
        if not self.seed_data.empty and not self.create_seed:
            print("Loop {} state: Agent {} hypothesizing".format(
                'initialization', self.agent.__class__.__name__))
            suggested_experiments = self.agent.get_hypotheses(
                self.candidate_data, self.seed_data)
        elif self.create_seed:
            np.random.seed(seed=random_state)
            _agent = RandomAgent(self.candidate_data, n_query=self.create_seed)
            print("Loop {} state: Agent {} hypothesizing".format(
                'initialization', _agent.__class__.__name__))
            suggested_experiments = _agent.get_hypotheses(self.candidate_data)
        else:
            raise ValueError(
                "No seed data available. Either supply or ask for creation.")

        print("Loop {} state: Running experiments".format(self.iteration))
        experiment_data = self.candidate_data.loc[suggested_experiments]
        self.job_status = self.experiment.submit(experiment_data)

        self.submitted_experiment_requests = suggested_experiments
        self.consumed_candidates = suggested_experiments
        self.create_seed = False
        self.initialized = True

        self.save("job_status")
        self.save("seed_data", method='pickle')
        self.save('submitted_experiment_requests')
        self.save("consumed_candidates")
        self.save("iteration")
        if self.s3_prefix:
            self.s3_sync()

    def initialize_with_icsd_seed(self, random_state=42):
        if self.initialized:
            raise ValueError(
                "Initialization may overwrite existing loop data. Exit.")
        cache_s3_objs(
            ["camd/shared-data/oqmd1.2_icsd_featurized_clean_v2.pickle"]
        )
        self.seed_data = pd.read_pickle(
            os.path.join(S3_CACHE, "camd/shared-data/oqmd1.2_icsd_featurized_clean_v2.pickle"))
        self.initialize(random_state=random_state)

    def report(self):
        # TODO: Can we use pandas here?
        with open(os.path.join(self.path, 'report.log'), 'a') as f:
            if self.iteration == 0:
                f.write("Iteration N_Discovery Total_Discovery N_candidates model-CV\n")
            report_string = "{:9} {:11} {:15} {:12} {:f}\n".format(
                self.iteration, np.sum(self.results_new_uids),
                np.sum(self.results_all_uids), len(self.candidate_data),
                self.agent.cv_score)
            f.write(report_string)

        self.generate_report_plot()

    def finalize(self):
        print("Finalizing campaign.")
        os.chdir(self.path)
        if self.finalizer:
            self.finalizer.finalize(self.path)
        if self.s3_prefix:
            self.s3_sync()

    @staticmethod
    def generate_report_plot(filename="report.png",
                             report_filename="report.log"):
        """
        Quick method for generating report plots

        Args:
            filename (str): output filename for plot to be saved
            report_filename (str): filename for the report to be read in

        Returns:
            (AxesSubplot) pyplot object corresponding to bar plot

        """
        # Generate plot
        data = pd.read_csv(report_filename, delim_whitespace=True)
        plt = pretty_plot(6, 4.5)
        ax = plt.gca()
        ax = data.plot(
            kind='bar', x='Iteration', y='Total_Discovery',
            legend=False, ax=ax
        )
        ax.set_ylabel("Total materials discovered")
        fig = ax.get_figure()
        if filename:
            fig.savefig(filename, dpi=70)

        # Close to avoid matplotlib memory warning
        plt.close()

    def load(self, data_holder, method='json', no_exist_fail=True):
        if method == 'pickle':
            m = pickle
            mode = 'rb'
        elif method == 'json':
            m = json
            mode = 'r'
        else:
            raise ValueError("Unknown data save method")

        file_name = os.path.join(self.path, data_holder+'.'+method)
        exists = os.path.exists(file_name)

        if exists:
            with open(file_name, mode) as f:
                self.__setattr__(data_holder, m.load(f))
        else:
            if no_exist_fail:
                raise IOError("No {} file exists".format(data_holder))
            else:
                self.__setattr__(data_holder, None)

    def save(self, data_holder, custom_name=None, method='json'):
        if custom_name:
            _path = os.path.join(self.path, custom_name)
        else:
            _path = os.path.join(self.path, data_holder+'.'+method)
        if method == 'pickle':
            m = pickle
            mode = 'wb'
        elif method == 'json':
            m = json
            mode = 'w'
        else:
            raise ValueError("Unknown data save method")
        with open(_path, mode) as f:
            m.dump(self.__getattribute__(data_holder), f)

        # Do s3 sync if present
        if self.s3_prefix:
            self.s3_sync()

    def get_state(self):
        return self.loop_state

    def s3_sync(self):
        """
        Syncs current run to s3_prefix and bucket
        """
        s3_sync(self.s3_bucket, self.s3_prefix, self.path)


# TODO: fix docstring
def loop_backup(src, new_dir_name):
    """
    Helper method to backup finished loop iterations.

    Args:
        src:
        new_dir_name:

    Returns:

    """
    os.mkdir(os.path.join(src, new_dir_name))
    _files = os.listdir(src)
    for file_name in _files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, new_dir_name)
