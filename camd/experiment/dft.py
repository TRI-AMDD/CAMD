# Copyright Toyota Research Institute 2019

import os
import uuid
import json
import re
import time
import shlex
import subprocess
import traceback
import warnings
from datetime import datetime

from monty.os import cd
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen import Composition
from camd.experiment.base import Experiment
from camd.utils.data import QMPY_REFERENCES, \
    QMPY_REFERENCES_HUBBARD


class OqmdDFTonMC1(Experiment):
    """
    An experiment class to manage Density Functional theory
    experiments on the MC1, an AWS-batch-based DFT-calculation
    system.
    """
    def __init__(self, poll_time=60, timeout=7200,
                 current_data=None, job_status=None):
        self.poll_time = poll_time
        self.timeout = timeout
        super().__init__(current_data=current_data,
                         job_status=job_status)

    def _update_job_status(self):
        """
        Updates the aggregate job status according to the
        entries of the current data's "status" column

        Returns:
            None

        """
        all_job_statuses = self.current_data['status']
        all_jobs_complete = all([status in ['SUCCEEDED', 'FAILED']
                                 for status in all_job_statuses])
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
        if self.job_status is not "COMPLETED":
            self.update_results()
        if self.job_status is not "COMPLETED":
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
        self.update_current_data(data)
        # Populate new columns
        new_columns = ['path', 'status', 'start_time', 'jobId',
                       'jobName', 'result', 'error', 'delta_e',
                       'elapsed_time']
        for new_column in new_columns:
            self.current_data[new_column] = None

        self.submit_dft_calcs_to_mc1()
        self.job_status = 'PENDING'
        return self.job_status

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
                structure_id, row["status"], row["elapsed_time"])
        print("Calc status:\n{}".format(status_string))
        print("Timeout is set as {}.".format(self.timeout))

    def monitor(self):
        """
        Method for continuously polling for completion


        Returns:
            (str): calculation status string

        """
        while self.job_status is not 'COMPLETED':
            time.sleep(self.poll_time)
            self.update_results()
            self.print_status()
            self.kill_lapsed_jobs()

        return self.job_status

    def submit_dft_calcs_to_mc1(self):
        """
        Helper method for submitting DFT calculations to MC1
        """
        tri_path = os.environ.get("TRI_PATH")
        if not tri_path:
            raise ValueError("TRI_PATH must be specified as env variable to "
                             "use camd MC1 interface")

        # Create run directory
        uuid_string = str(uuid.uuid4()).replace('-', '')
        parent_dir = os.path.join(tri_path, "model", "oqmdvasp", "3",
                                  "u", "camd", "run{}".format(uuid_string))
        if any(['_' in value for value in self.current_data.index]):
            raise ValueError("Structure keys cannot contain underscores for "
                             "mc1 compatibility")

        for structure_id, row in self.current_data.iterrows():
            calc_path = os.path.join(parent_dir, structure_id, "_1")
            os.makedirs(calc_path)
            with cd(calc_path):
                # Write input cif file and python model file
                row['structure'].to(filename="POSCAR")
                with open("model.py", "w") as f:
                    f.write(MODEL_TEMPLATE)

                # Submit to mc1
                # TODO: ensure this is checked for failure to submit
                print("Submitting job: {}".format(structure_id))
                try:
                    response = subprocess.check_output(
                        ["trisub", "-q", "oqmd_test_queue",
                         "-r", "16000", "-c", "16", "-g", "us-east-1"]
                    )
                except subprocess.CalledProcessError as e:
                    print(e.output)

                response = response.decode('utf-8')
                response = re.findall("({.+})", response, re.DOTALL)[0]
                data = json.loads(response)
                data.update({"path": os.getcwd(),
                             "status": "SUBMITTED",
                             "start_time": datetime.utcnow()})
                update_dataframe_row(self.current_data, structure_id, data)

    def check_dft_calcs(self):
        """
        Helper function to check DFT calculations via polling
        AWS batch.  Updates current data with latest AWS batch
        response.
        """
        for structure_id, calc in self.current_data.iterrows():
            if calc['status'] in ['SUCCEEDED', 'FAILED']:
                continue
            path = calc['path']
            print("Checking status of {}: {}".format(path, structure_id))
            aws_cmd = "aws batch describe-jobs --jobs " \
                      "--region=us-east-1 {}".format(calc['jobId'])
            result = subprocess.check_output(shlex.split(aws_cmd))
            result = json.loads(result)
            aws_status = result["jobs"][0]["status"]
            if aws_status == "SUCCEEDED":
                os.chdir(path)
                subprocess.call('trisync')
                os.chdir('simulation')
                try:
                    vr = Vasprun('static/vasprun.xml')
                    vr_dict = vr.as_dict()
                    delta_e = get_qmpy_formation_energy(
                        vr_dict['output']['final_energy_per_atom'],
                        vr_dict['pretty_formula'], 1)
                    data = {"status": "SUCCEEDED", "error": None, "result": vr,
                            "delta_e": delta_e}
                except Exception as e:
                    error_doc = {}
                    with open('err') as errfile:
                        error_doc.update({"trisub_stderr": errfile.read()})
                    error_doc.update({"camd_exception": "{}".format(e),
                                      "camd_traceback": traceback.format_exc()})

                    # Dump error docs to avoid Pandas issues with dict values
                    data = {"status": "FAILED", "error": json.dumps(error_doc)}

            elif aws_status == "FAILED":
                error_doc = {"aws_fail": result['jobs'][0]['attempts'][-1]['statusReason']}
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
        running_jobs = self.current_data[self.current_data['status'] == 'RUNNING']
        lapsed_jobs = running_jobs[running_jobs['elapsed_time'] > self.timeout]

        # Kill AWS job
        for structure_id, row in lapsed_jobs.iterrows():
            kill_cmd = "aws batch terminate-job --region=us-east-1 " \
                       "--job-id {} --reason camd_timeout".format(row['jobId'])
            kill_result = subprocess.check_output(shlex.split(kill_cmd))
            self.current_data.loc[structure_id, 'status'] = 'FAILED'
            self.current_data.loc[structure_id, 'error'] = 'timeout'
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
    # Kids, don't try this at home
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
        if (element in QMPY_REFERENCES_HUBBARD) \
                and ('O' in composition):
            energy -= QMPY_REFERENCES_HUBBARD[element] * weight
    return energy


def update_dataframe_row(dataframe, index, update_dict):
    """
    Method to update a dataframe row via an update_dictionary
    and an index, similarly to Dict.update()

    Args:
        dataframe (DataFrame): DataFrame for which rows should
            be updated
        index: index value for dataframe
        update_dict ({}): update dictionary for dataframe

    Returns:
        None

    """
    for key, value in update_dict.items():
        dataframe.loc[index, key] = value
