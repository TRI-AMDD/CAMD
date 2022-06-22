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
from monty.os import cd
from monty.tempfile import ScratchDir
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.composition import Composition
from camd.experiment.base import Experiment
from camd.utils.data import QMPY_REFERENCES, QMPY_REFERENCES_HUBBARD, \
    get_chemsys, get_common_prefixes


class OqmdDFTonMC1(Experiment):
    """
    An experiment class to manage Density Functional theory
    experiments on the MC1, an AWS-batch-based DFT-calculation
    system.
    """

    def __init__(self, poll_time=60, timeout=7200, current_data=None, job_status=None,
                 cleanup_directories=True, container_version="oqmdvasp/3",
                 batch_queue="oqmd_test_queue", use_cached=False,
                 prefix_append=None):
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

        container, version = container_version.split('/')
        parent_dir = os.path.join(
            tri_path,
            "model",
            container,
            version,
            "u",
            "camd",
        )
        if prefix_append is not None:
            if '_' in prefix_append:
                raise ValueError("Prefix cannot contain underscores for mc1 compatibility")
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
            sim_path = row['path'].replace("model", "simulation")
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
        s3_parent = self.parent_dir.replace('model', 'simulation')
        s3_parent = s3_parent.replace(tri_path + '/', "")
        cached_experiments = pd.DataFrame()
        # Get all experiment folders
        chemsyses = set([get_chemsys(s) for s in candidate_data['structure']])
        experiment_dirs = []
        for chemsys in chemsyses:
            chemsys_dirs = get_common_prefixes(
                tri_bucket,
                os.path.join(s3_parent, chemsys)
            )
            experiment_dirs.extend(chemsys_dirs)
        for structure_id, row in tqdm(candidate_data.iterrows(), total=len(candidate_data)):
            if not structure_id.replace('-', '') in experiment_dirs:
                continue
            calc_path = os.path.join(
                s3_parent, get_chemsys(row['structure']),
                structure_id.replace('-', ''), "_1/")
            with ScratchDir('.'):
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
                            Bucket=tri_bucket, Key=os.path.join(calc_path, 'err'))
                        errtxt = err_obj['Body'].read().decode('utf-8')
                        error_doc.update(
                            {"mc1_stderr": errtxt}
                        )
                    except ClientError:
                        print('No error file for {}'.format(calc_path))
                    error_doc.update(
                        {
                            "camd_exception": "{}".format(e),
                            "camd_traceback": traceback.format_exc(),
                        }
                    )
                    # Dump error docs to avoid Pandas issues with dict values
                    data = {"status": "FAILED", "error": json.dumps(error_doc)}
                update_dataframe_row(cached_experiments, structure_id, data, add_columns=True)
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
                self.parent_dir, get_chemsys(row['structure']),
                structure_id.replace('-', ''), "_1")
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
                    data = {"path": os.getcwd(),
                            "status": "FAILED",
                            "error": "failed submission {}".format(e.output)}

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


def update_dataframe_row(dataframe, index, update_dict,
                         add_columns=False):
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
        dataframe.loc[index, key] = value
