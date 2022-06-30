# Copyright Toyota Research Institute 2019
"""
Module containing DFT-related experiments, typically
to be run asynchronously with a campaign.
"""

import os
import json
import re
import time
import shlex
import shutil
import subprocess
import traceback
import warnings
from datetime import datetime

import boto3
from botocore.errorfactory import ClientError
from tqdm import tqdm
import pandas as pd
import numpy as np
from monty.os import cd
from monty.tempfile import ScratchDir
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Composition, Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from camd.experiment.base import Experiment
from camd.utils.data import (
    MP_REFERENCES,
    QMPY_REFERENCES,
    QMPY_REFERENCES_HUBBARD,
    get_chemsys,
    get_common_prefixes,
)

from atomate.vasp.database import VaspCalcDb
from atomate.vasp.config import (
    ADD_WF_METADATA,
    DB_FILE,
    VASP_CMD,
)
from atomate.vasp.powerups import (
    add_common_powerups,
    add_wf_metadata,
)
from atomate.vasp.workflows.base.core import get_wf

from fireworks.core.rocket_launcher import rapidfire


def wf_structure_optimization(structure, wf_name=None, c=None):
    """
    Hacking the atomate wf_structure optimization to allow
    wf_name added to the metadata

    Args:
        structure (Structure)
        wf_name (str)
        c (dict)
    Returns:

    """
    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)
    user_incar_settings = c.get("USER_INCAR_SETTINGS")

    wf = get_wf(
        structure,
        "optimize_only.yaml",
        vis=MPRelaxSet(
            structure, force_gamma=True, user_incar_settings=user_incar_settings
        ),
        common_params={"vasp_cmd": vasp_cmd, "db_file": db_file},
        wf_metadata={"wf_name": wf_name},
    )

    wf = add_common_powerups(wf, c)

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf


class AtomateExperiment(Experiment):
    """
    A class for brokering atomate experiments
    TODO: 1. formation energy solution
    """

    def __init__(
        self,
        launchpad,
        db_file,
        fworker=None,
        atomate_workflow=wf_structure_optimization,
        nlaunches=0,
        max_loops=-1,
        sleep_time=5,
        m_dir=None,
        poll_time=60,
        current_data=None,
        job_status=None,
        history=None,
        launch_from_local=False,
    ):
        """
        Initializes an atomate experiment.

        Args:
            launchpad (LaunchPad): launchpad
            db_file (str): path to atomate db config file
            fworker (FWorker object): fworker
            atomate_workflow (): atomate workflow, default the optimziation wf
            nlaunches (int): 0 means 'until completion', -1 or "infinite" means to loop until max_loops
            max_loops (int): maximum number of loops (default -1 is infinite)
            sleep_time (int): secs to sleep between rapidfire loop iterations
            m_dir (str): the directory in which to loop Rocket running
            poll_time (int): time in seconds to wait in between queries of db
            current_data (pandas.DataFrame): dataframe corresponding to
                currently submitted experiments
            job_status (str): status of the experiment, PENDING or COMPLETED
            history (pandas.DataFrame): history of past experiments
            launch_from_local (bool): whether to launch from local
        """
        self.current_data = current_data
        self.job_status = job_status
        self._history = history or []
        self.launchpad = launchpad
        self.fworker = fworker
        self.nlaunches = nlaunches
        self.max_loops = max_loops
        self.sleep_time = sleep_time
        self.m_dir = m_dir
        self.db = VaspCalcDb.from_db_file(db_file).db
        self.wf = atomate_workflow
        self.poll_time = poll_time
        self.launch_from_local = launch_from_local

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

    def get_results(self):
        """
        Gets the results from the current run.

        Returns:
            (DataFrame): dataframe corresponding to the
                current set of results

        """
        if self.job_status != "COMPLETED":
            self.update_results()
        if self.job_status != "COMPLETED":
            warnings.warn("Some calculations have not finished")
        return self.current_data

    def monitor(self):
        """
        Method for continuously polling for completion
        Returns:
            (str): calculation status string
        """
        while self.job_status != "COMPLETED":
            time.sleep(self.poll_time)
            self.update_results()
            self.print_status()
        return self.job_status

    def update_results(self):
        """
        Update the current data with the latest wf, fw, launch, task information
        """
        task_id = None
        launch_id = None
        task_status = None
        output = None
        input = None
        task_dir = None
        calcs_reversed = None
        delta_e = None
        for structure_id, row in self.current_data.iterrows():
            wf_entry = self.db.workflows.find_one(
                {
                    "metadata.wf_name": row["wf_name"],
                    "fw_states.{}".format(row["fw_id"]): {"$exists": True},
                }
            )
            fw_entry = self.db.fireworks.find_one({"fw_id": row["fw_id"]})
            if len(fw_entry["launches"]) > 0:
                launch_id = fw_entry["launches"][-1]
                launch_entry = self.db.launches.find_one({"launch_id": launch_id})
                if launch_entry["action"]:
                    if "task_id" in launch_entry["action"]["stored_data"]:
                        task_id = launch_entry["action"]["stored_data"]["task_id"]
                        task_entry = self.db.tasks.find_one({"task_id": task_id})
                        task_status = task_entry["state"]
                        output = task_entry["output"]
                        input = task_entry["input"]
                        task_dir = task_entry["dir_name"]
                        calcs_reversed = task_entry["calcs_reversed"]
                        if task_status == "successful":
                            delta_e = self.get_delta_e(task_entry)
            update_data = {
                "task_id": task_id if task_id else None,
                "launch_id": launch_id if launch_id else None,
                "wf_status": wf_entry["state"],
                "task_status": task_status if task_status else None,
                "output": output if output else None,
                "input": input if input else None,
                "task_dir": task_dir if task_dir else None,
                "calcs_reversed": json.dumps(calcs_reversed)
                if calcs_reversed
                else None,
                "final_structure": output["structure"] if output else None,
                "final_energy_per_atom": output["energy_per_atom"] if output else None,
                "delta_e": delta_e,
            }
            update_dataframe_row(self.current_data, structure_id, update_data)
        self._update_job_status()

    def _update_job_status(self):
        """
        Update the job_status flag by checking all wf status
        job_status is "COMPLETED" only when all wf_status are COMPLETED/FIZZLED
        """
        wf_status = self.current_data["wf_status"]
        if np.all([i in ["COMPLETED", "FIZZLED"] for i in wf_status]):
            self.job_status = "COMPLETED"

    def print_status(self):
        """
        Prints current status of experiment according to
        the data in the current_data attribute
        """
        status_string = ""
        for structure_id, row in self.current_data.iterrows():
            status_string += "{}: {}\n".format(structure_id, row["wf_status"])
        print("Calc status:\n{}".format(status_string))
        # print("Timeout is set as {}.".format(self.timeout))

    def submit(self, data):
        """
        Args:
            data (DataFrame): dataframe containing all necessary
                data to conduct the experiment(s).  May be one
                row, may be multiple rows

        Returns:
            None
        """
        self.update_current_data(data)
        new_columns = [
            "wf_spec",
            "wf_name",
            "fw_id",
            "task_id",
            "launch_id",
            "wf_status",  # status of the workflow (ready, waiting, running, fizzeld, completed)
            "task_status",  # status of the task (successful, failed)
            "output",
            "input",
            "task_dir",
            "calcs_reversed",
            "final_structure",
            "final_energy_per_atom",
            "delta_e",  # formation energy per atom
        ]
        for new_column in new_columns:
            self.current_data[new_column] = None
        self.add_wfs()
        if self.launch_from_local:
            self.launch()
        self.job_status = "PENDING"
        return self.job_status

    def add_wfs(self):
        """
        Helper function to add workflows to the launchpad
        and update the wf_name, fw_ids
        """
        for structure_id, row in self.current_data.iterrows():
            wf_name = f"opt_{structure_id}"
            wf = wf_structure_optimization(row["structure"], wf_name=wf_name)
            fw_ids = self.launchpad.add_wf(wf)
            launch_info = {
                "wf_spec": wf,
                "wf_name": wf_name,
                "fw_id": sorted(list(fw_ids.values()))[-1],
            }
            update_dataframe_row(self.current_data, structure_id, launch_info)

    def launch(self):
        """
        Helper method for launching firework
        """
        rapidfire(
            self.launchpad,
            self.fworker,
            nlaunches=self.nlaunches,
            max_loops=self.max_loops,
            sleep_time=self.sleep_time,
            m_dir=self.m_dir,
        )

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

    def get_delta_e(self, task_entry):
        """
        Helper function to get the formation energy from task entry
        Args:
            task_entry (dict): tasks entry
        Returns:
            formation energy (float)
        """
        total_e = task_entry["output"]["energy"]
        composition = Structure.from_dict(task_entry["output"]["structure"]).composition
        psp = task_entry["input"]["pseudo_potential"]
        functional = psp["functional"].capitalize()
        labels = psp["labels"]
        potcar_symbols = [" ".join([functional, i]) for i in labels]
        hubbards = task_entry["input"]["hubbards"]
        return get_mp_formation_energy(
            total_e, composition.__str__(), potcar_symbols, hubbards
        )


class OqmdDFTonMC1(Experiment):
    """
    An experiment class to manage Density Functional theory
    experiments on the MC1, an AWS-batch-based DFT-calculation
    system.
    """

    def __init__(
        self,
        poll_time=60,
        timeout=7200,
        current_data=None,
        job_status=None,
        cleanup_directories=True,
        container_version="oqmdvasp/3",
        batch_queue="oqmd_test_queue",
        use_cached=False,
        prefix_append=None,
    ):
        """
        Initializes an OqmdDFTonMC1 instance

        Args:
            poll_time (int): time in seconds to wait in between queries of aws batch
            timeout (int): time in seconds to wait before killing batch jobs
            current_data (pandas.DataFrame): dataframe corrsponding to current data
            job_status (str): job status
            cleanup_directories (bool): flag to enable cleaning up of DFT results after
                runs are over
            container_version (str): container for mc1, e.g. "oqmdvasp/3" or "gpaw/1" which
                dictates where things will be run
            batch_queue (str): name of aws batch queue to submit to
            use_cached (bool): whether to used cached results from prior campaigns
            prefix_append (str): appended prefix to use for caching and data organization,
                e.g. proto-dft-1
        """
        self.poll_time = poll_time
        self.timeout = timeout
        self.cleanup_directories = cleanup_directories
        self.use_cached = use_cached
        self.batch_queue = batch_queue

        # Build parent directory using TRI_PATH and container/version
        tri_path = os.environ.get("TRI_PATH")
        tri_bucket = os.environ.get("TRI_BUCKET")
        if not (tri_path and tri_bucket):
            raise ValueError(
                "TRI_PATH and TRI_BUCKET must be specified as env "
                "variable to use camd MC1 interface"
            )

        container, version = container_version.split("/")
        parent_dir = os.path.join(
            tri_path,
            "model",
            container,
            version,
            "u",
            "camd",
        )
        if prefix_append is not None:
            if "_" in prefix_append:
                raise ValueError(
                    "Prefix cannot contain underscores for mc1 compatibility"
                )
            parent_dir = os.path.join(parent_dir, prefix_append)
        self.parent_dir = parent_dir

        super().__init__(current_data=current_data, job_status=job_status)

    def _update_job_status(self):
        """
        Updates the aggregate job status according to the
        entries of the current data's "status" column

        Returns:
            None

        """
        all_job_statuses = self.current_data["status"]
        all_jobs_complete = all(
            [status in ["SUCCEEDED", "FAILED"] for status in all_job_statuses]
        )
        if all_jobs_complete:
            self.job_status = "COMPLETED"

    def update_results(self):
        """
        Updates the results by checking the AWS Batch system
        via check_dft_calcs

        Returns:
            None

        """
        # Update data
        self.check_dft_calcs()

        # Update time
        elapsed_time = datetime.utcnow() - self.current_data["start_time"]
        elapsed_time = [e.total_seconds() for e in elapsed_time]
        self.current_data["elapsed_time"] = elapsed_time

        # Update job status
        self._update_job_status()

    def get_results(self):
        """
        Gets the results from the current run.

        Returns:
            (DataFrame): dataframe corresponding to the
                current set of results

        """
        if self.job_status != "COMPLETED":
            self.update_results()
        if self.job_status != "COMPLETED":
            warnings.warn("Some calculations have not finished")
        return self.current_data

    def submit(self, data):
        """
        Args:
            data (DataFrame): dataframe representing structure
                inputs to experiment.  This dataframe must have
                a 'structure' column

        Returns:
            (str): string corresponding to job status

        """
        if self.current_data is not None and self.cleanup_directories:
            self.clean_current_paths()
        self.update_current_data(data)
        # Populate new columns
        new_columns = [
            "path",
            "status",
            "start_time",
            "jobId",
            "jobName",
            "jobArn",
            "result",
            "error",
            "delta_e",
            "elapsed_time",
        ]
        for new_column in new_columns:
            self.current_data[new_column] = None

        self.submit_dft_calcs_to_mc1()
        self.job_status = "PENDING"
        return self.job_status

    def clean_current_paths(self):
        """
        Helper method to clean simulation results in current paths

        Returns:
            None

        """
        for idx, row in self.current_data.iterrows():
            # Clean simulation directories
            sim_path = row["path"].replace("model", "simulation")
            if os.path.exists(sim_path):
                shutil.rmtree(sim_path)

    def print_status(self):
        """
        Prints current status of experiment according to
        the data in the current_data attribute

        Returns:
            None

        """
        status_string = ""
        for structure_id, row in self.current_data.iterrows():
            status_string += "{}: {} {}\n".format(
                structure_id, row["status"], row["elapsed_time"]
            )
        print("Calc status:\n{}".format(status_string))
        print("Timeout is set as {}.".format(self.timeout))

    def monitor(self):
        """
        Method for continuously polling for completion


        Returns:
            (str): calculation status string

        """
        while self.job_status != "COMPLETED":
            time.sleep(self.poll_time)
            self.update_results()
            self.print_status()
            self.kill_lapsed_jobs()

        return self.job_status

    def fetch_cached(self, candidate_data):
        """
        Fetches cached data based on candidate data

        Args:
            candidate_data (pd.DataFrame): Pandas dataframe

        Returns:
            (pandas.DataFrame) dataframe with data filled out

        """
        tri_path = os.environ.get("TRI_PATH")
        tri_bucket = os.environ.get("TRI_BUCKET")
        s3_client = boto3.client("s3")
        # Scrub tri_path and replace model with simulation
        # to get s3 key
        s3_parent = self.parent_dir.replace("model", "simulation")
        s3_parent = s3_parent.replace(tri_path + "/", "")
        cached_experiments = pd.DataFrame()
        # Get all experiment folders
        chemsyses = set([get_chemsys(s) for s in candidate_data["structure"]])
        experiment_dirs = []
        for chemsys in chemsyses:
            chemsys_dirs = get_common_prefixes(
                tri_bucket, os.path.join(s3_parent, chemsys)
            )
            experiment_dirs.extend(chemsys_dirs)
        for structure_id, row in tqdm(
            candidate_data.iterrows(), total=len(candidate_data)
        ):
            if not structure_id.replace("-", "") in experiment_dirs:
                continue
            calc_path = os.path.join(
                s3_parent,
                get_chemsys(row["structure"]),
                structure_id.replace("-", ""),
                "_1/",
            )
            with ScratchDir("."):
                # Figure out whether prior submission exists
                cached_experiments = cached_experiments.append(row)
                # TODO: figure out whether file exists in s3
                # TODO: this is a little crude, could use boto3
                try:
                    # import pdb; pdb.set_trace()
                    vr_path = os.path.join(calc_path, "static", "vasprun.xml")
                    cmd = "aws s3 cp s3://{}/{} .".format(tri_bucket, vr_path)
                    subprocess.call(shlex.split(cmd))
                    vr = Vasprun("vasprun.xml")
                    vr_dict = vr.as_dict()
                    delta_e = get_qmpy_formation_energy(
                        vr_dict["output"]["final_energy_per_atom"],
                        vr_dict["pretty_formula"],
                        1,
                    )
                    data = {
                        "status": "SUCCEEDED",
                        "error": None,
                        "result": vr,
                        "delta_e": delta_e,
                    }
                except Exception as e:
                    error_doc = {}
                    try:
                        err_obj = s3_client.get_object(
                            Bucket=tri_bucket, Key=os.path.join(calc_path, "err")
                        )
                        errtxt = err_obj["Body"].read().decode("utf-8")
                        error_doc.update({"mc1_stderr": errtxt})
                    except ClientError:
                        print("No error file for {}".format(calc_path))
                    error_doc.update(
                        {
                            "camd_exception": "{}".format(e),
                            "camd_traceback": traceback.format_exc(),
                        }
                    )
                    # Dump error docs to avoid Pandas issues with dict values
                    data = {"status": "FAILED", "error": json.dumps(error_doc)}
                update_dataframe_row(
                    cached_experiments, structure_id, data, add_columns=True
                )
        return cached_experiments

    def submit_dft_calcs_to_mc1(self):
        """
        Helper method for submitting DFT calculations to MC1
        """

        # Create run directory
        # uuid_string = str(uuid.uuid4()).replace("-", "")

        if any(["_" in value for value in self.current_data.index]):
            raise ValueError(
                "Structure keys cannot contain underscores for " "mc1 compatibility"
            )

        for structure_id, row in self.current_data.iterrows():
            # Replace structure id in path to avoid confusing mc1
            calc_path = os.path.join(
                self.parent_dir,
                get_chemsys(row["structure"]),
                structure_id.replace("-", ""),
                "_1",
            )
            os.makedirs(calc_path, exist_ok=True)
            with cd(calc_path):
                # Write input cif file and python model file
                row["structure"].to(filename="POSCAR")
                with open("model.py", "w") as f:
                    f.write(MODEL_TEMPLATE)

                # Submit to mc1
                # TODO: ensure this is checked for failure to submit
                print("Submitting job: {}".format(structure_id))
                try:
                    response = subprocess.check_output(
                        [
                            "trisub",
                            "-q",
                            self.batch_queue,
                            "-r",
                            "16000",
                            "-c",
                            "16",
                            "-g",
                            "us-east-1",
                        ]
                    )
                    response = response.decode("utf-8")
                    response = re.findall("({.+})", response, re.DOTALL)[0]
                    data = json.loads(response)
                    data.update(
                        {
                            "path": os.getcwd(),
                            "status": "SUBMITTED",
                            "start_time": datetime.utcnow(),
                        }
                    )
                except subprocess.CalledProcessError as e:
                    print(e.output)
                    data = {
                        "path": os.getcwd(),
                        "status": "FAILED",
                        "error": "failed submission {}".format(e.output),
                    }

                update_dataframe_row(self.current_data, structure_id, data)

    def check_dft_calcs(self):
        """
        Helper function to check DFT calculations via polling
        AWS batch.  Updates current data with latest AWS batch
        response.
        """
        for structure_id, calc in self.current_data.iterrows():
            if calc["status"] in ["SUCCEEDED", "FAILED"]:
                continue
            path = calc["path"]
            print("Checking status of {}: {}".format(path, structure_id))
            aws_cmd = "aws batch describe-jobs --jobs " "--region=us-east-1 {}".format(
                calc["jobId"]
            )
            result = subprocess.check_output(shlex.split(aws_cmd))
            result = json.loads(result)
            aws_status = result["jobs"][0]["status"]
            if aws_status == "SUCCEEDED":
                os.chdir(path)
                subprocess.call("trisync")
                os.chdir("simulation")
                try:
                    vr = Vasprun("static/vasprun.xml")
                    vr_dict = vr.as_dict()
                    delta_e = get_qmpy_formation_energy(
                        vr_dict["output"]["final_energy_per_atom"],
                        vr_dict["pretty_formula"],
                        1,
                    )
                    data = {
                        "status": "SUCCEEDED",
                        "error": None,
                        "result": vr,
                        "delta_e": delta_e,
                    }
                except Exception as e:
                    error_doc = {}
                    if os.path.isfile("err"):
                        with open("err") as errfile:
                            error_doc.update({"mc1_stderr": errfile.read()})
                    error_doc.update(
                        {
                            "camd_exception": "{}".format(e),
                            "camd_traceback": traceback.format_exc(),
                        }
                    )

                    # Dump error docs to avoid Pandas issues with dict values
                    data = {"status": "FAILED", "error": json.dumps(error_doc)}
                os.chdir(path)

            elif aws_status == "FAILED":
                error_doc = {
                    "aws_fail": result["jobs"][0]["attempts"][-1]["statusReason"]
                }
                data = {"status": "FAILED", "error": json.dumps(error_doc)}
            else:
                data = {"status": aws_status}
            update_dataframe_row(self.current_data, structure_id, data)

    def kill_lapsed_jobs(self):
        """
        Method for killing lapsed jobs according to the amount
        of elapsed time and the object's timeout attribute

        Returns:
            None
        """
        running_jobs = self.current_data[self.current_data["status"] == "RUNNING"]
        lapsed_jobs = running_jobs[running_jobs["elapsed_time"] > self.timeout]

        # Kill AWS job
        for structure_id, row in lapsed_jobs.iterrows():
            kill_cmd = (
                "aws batch terminate-job --region=us-east-1 "
                "--job-id {} --reason camd_timeout".format(row["jobId"])
            )
            kill_result = subprocess.check_output(shlex.split(kill_cmd))
            print("{} job killed: ".format(kill_result))
            self.current_data.loc[structure_id, "status"] = "FAILED"
            self.current_data.loc[structure_id, "error"] = "timeout"
        self._update_job_status()


MODEL_TEMPLATE = """
import os

import qmpy
from qmpy.materials.structure import Structure
from qmpy.analysis.vasp.calculation import Calculation
from qmpy import io
import time


# TODO: definitely move this somewhere else, as it's not
#       meant to be imported
def run_oqmd_calculation(poscar_filename):
    starting_structure = io.poscar.read(poscar_filename)

    # Relaxation
    os.mkdir("relax")
    os.chdir("relax")
    calc = Calculation()
    calc.setup(starting_structure, "relaxation")
    os.system("mpirun -n 16 vasp_std")
    # Just in case the mysql server process dies
    os.system("sudo -u mysql mysqld &")
    relaxed_structure = io.poscar.read("CONTCAR")
    os.chdir('..')

    # Relaxation
    os.mkdir("static")
    os.chdir("static")
    calc = Calculation()
    calc.setup(relaxed_structure, "static")
    os.system("mpirun -n 16 vasp_std")
    os.chdir('..')


if __name__ == '__main__':
    run_oqmd_calculation("POSCAR")
"""


def get_qmpy_formation_energy(total_e, formula, n_atoms):
    """
    Helper function to computer qmpy-compatible formation
    energy using reference energies extracted from OQMD

    Args:
        total_e (float): total energy
        formula (str): chemical formula
        n_atoms (int): number of atoms

    Returns:
        (float): qmpy-compatible formation energy

    """
    composition = Composition(formula).fractional_composition
    energy = total_e / n_atoms
    for element, weight in composition.as_dict().items():
        energy -= QMPY_REFERENCES[element] * weight
        if (element in QMPY_REFERENCES_HUBBARD) and ("O" in composition):
            energy -= QMPY_REFERENCES_HUBBARD[element] * weight
    return energy


def get_mp_formation_energy(
    total_e, formula, potcar_symbols, hubbards={}, explain=False
):
    """
    Helper function to computer mp-compatible formation
    energy using reference energies extracted from MP

    Args:
        total_e (float): total energy (uncorrected)
        formula (str): chemical formula
        potcar_symbols (list): list of potcar symbols
        hubbards (dict): hubbard value, if none, the
                        run_type = 'GGA'
                        else run_type = 'GGA+U'
        explain (bool): whether to print out the explanation
                        of the correction

    Returns:
        (float): mp-compatible formation energy (eV/atom)

    """
    compatibility = MaterialsProjectCompatibility()
    comp = Composition(formula)
    run_type = "GGA+U" if hubbards else "GGA"
    is_hubbard = True if hubbards else False
    entry = ComputedEntry(
        composition=comp,
        energy=total_e,
        parameters={
            "potcar_symbols": potcar_symbols,
            "run_type": run_type,
            "hubbards": hubbards,
            "is_hubbard": is_hubbard,
        },
    )
    entry = compatibility.process_entry(entry)
    if explain:
        print(compatibility.explain(entry))
    energy = entry.energy
    for el, occ in comp.items():
        energy -= MP_REFERENCES[el.name] * occ
    return energy / comp.num_atoms


def update_dataframe_row(dataframe, index, update_dict, add_columns=False):
    """
    Method to update a dataframe row via an update_dictionary
    and an index, similarly to Dict.update()

    Args:
        dataframe (DataFrame): DataFrame for which rows should
            be updated
        index: index value for dataframe
        update_dict ({}): update dictionary for dataframe
        add_columns (bool): whether to add non-existent
            columns to the dataframe

    Returns:
        None

    """
    for key, value in update_dict.items():
        if add_columns and key not in dataframe.columns:
            dataframe[key] = None
        if isinstance(value, dict):
            dataframe.loc[index, key] = json.dumps(value)
        else:
            dataframe.loc[index, key] = value
