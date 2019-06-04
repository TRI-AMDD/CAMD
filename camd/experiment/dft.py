# Copyright Toyota Research Institute 2019

import os
import uuid
import json
import re
import time
from monty.os import cd
from monty.tempfile import ScratchDir
import shlex
from pymatgen.io.vasp.outputs import Vasprun
import subprocess
import traceback
import warnings


from camd.experiment.base import Experiment


class OqmdDFTonMC1(Experiment):
    def __init__(self, params):

        if 'structure_dict' not in params:
            raise ValueError("A dictionary of structures must be provided as input as structure_dict")

        self.structure_dict = params['structure_dict']
        self.poll_time = params['poll_time'] if 'poll_time' in params else 60
        self.timeout = params['timeout'] if 'timeout' in params else 7200

        self.unique_ids = params['unique_ids'] if 'unique_ids' in params else []
        self.job_status = params['job_status'] if 'job_status' in params else {}
        super().__init__(params)

    def start(self):
        """No start procedure"""
        pass

    def get_state(self):
        self.job_status = check_dft_calcs(self.job_status)
        return all([doc['status'] in ['SUCCEEDED', 'FAILED']
                    for doc in self.job_status.values()])

    def get_results(self, indices):
        # This gets the formation energies.
        if not self.get_state():
            warnings.warn("Some calculations did not finish.")
        results_dict = {}
        for structure_id, calc in self.job_status.items():
            if calc['status'] == "SUCCEEDED":
                results_dict['structure_id'] = calc['result']
        return results_dict

    def submit(self, unique_ids=None):
        """
        Args:
            unique_ids (list): Unique ids for structures to run from the structure_dict. If None, all entries in
            structure_dict are submitted.
        Returns:

        """
        self.unique_ids = unique_ids if unique_ids else list(self.structure_dict.keys())
        self._structures_to_run = dict([(k, v) for k, v in self.structure_dict.items() if k in self.unique_ids])
        self.job_status = submit_dft_calcs_to_mc1(self._structures_to_run)
        return self.job_status

    def run_monitor(self):
        """
        Returns:
            (dict): calculation status, including results
        """
        finished = self.get_state()
        with ScratchDir("."):
            while not finished:
                time.sleep(self.poll_time)
                finished = self.get_state()
                status_string = "\n".join(["{}: {}".format(key, value["status"])
                                           for key, value in self.job_status.items()])
                print("Calc status:\n{}".format(status_string))

                for doc in self.job_status.values():
                    doc["elapsed_time"] = time.time() - doc["start_time"]
                    if doc["elapsed_time"] > self.timeout:
                        if doc['status'] not in ['SUCCEEDED', 'FAILED']:
                            # Update job status to reflect timeout
                            doc.update({"status": "FAILED",
                                        "error": "timeout"})
                            # Kill AWS job
                            kill_cmd = "aws batch terminate-job --job-id {} --reason camd_timeout".format(
                                doc['jobId'])
                            kill_result = subprocess.check_output(shlex.split(kill_cmd))
        return self.job_status

    @classmethod
    def from_jon_status(cls, params, job_status):
        params["job_status"] = job_status
        params["unique_ids"] = list(job_status.keys())
        return cls(params)


def submit_dft_calcs_to_mc1(structure_dict):
    """
    Placeholder function for fetching DFT calculations from MC1
    using oqmd container

    Args:
        structure_dict (dict): dictionary of structures keyed by
            some string identifier
    """
    starting_dir = os.getcwd()
    tri_path = os.environ.get("TRI_PATH")
    if not tri_path:
        raise ValueError("TRI_PATH must be specified as env variable to "
                         "use camd MC1 interface")
    # Create run directory
    uuid_string = str(uuid.uuid4()).replace('-', '')
    parent_dir = os.path.join(tri_path, "model", "oqmdvasp", "2",
                              "u", "camd", "run{}".format(uuid_string))
    if any(['_' in key for key in structure_dict.keys()]):
        raise ValueError("Structure keys cannot contain underscores for "
                         "mc1 compatibility")

    calc_status = {}
    for structure_id, structure in structure_dict.items():
        calc_path = os.path.join(parent_dir, structure_id, "_1")
        os.makedirs(calc_path)
        with cd(calc_path):
            # Write input cif file and python model file
            structure.to(filename="POSCAR")
            with open("model.py", "w") as f:
                f.write(MODEL_TEMPLATE)

            # Submit to mc1
            # TODO: ensure this is checked for failure to submit
            print("Submitting job: {}".format(structure_id))
            calc = subprocess.check_output(["trisub", "-q", "small"])
            calc = calc.decode('utf-8')
            calc = re.findall("({.+})", calc, re.DOTALL)[0]
            calc = json.loads(calc)
            calc.update({"path": os.getcwd(),
                         "status": "SUBMITTED",
                         "start_time": time.time()})
            calc_status[structure_id] = calc

    return calc_status


def check_dft_calcs(calc_status):
    """

    Args:
        calc_status (dict):

    Returns:
        updated calc_status dictionary

    """
    for structure_id, calc in calc_status.items():
        if calc['status'] in ['SUCCEEDED', 'FAILED']:
            continue
        path = calc['path']
        print("Checking status of {}: {}".format(path, structure_id))
        aws_cmd = "aws batch describe-jobs --jobs {}".format(calc['jobId'])
        result = subprocess.check_output(shlex.split(aws_cmd))
        result = json.loads(result)
        aws_status = result["jobs"][0]["status"]
        if aws_status == "SUCCEEDED":
            os.chdir(path)
            subprocess.call('trisync')
            os.chdir('simulation')
            try:
                vr = Vasprun('static/vasprun.xml')
                calc.update({
                    "status": "SUCCEEDED",
                    "error": None,
                    "result": vr.as_dict()
                })
            except Exception as e:
                error_doc = {}
                with open('err') as errfile:
                    error_doc.update({"trisub_stderr": errfile.read()})
                error_doc.update({"camd_exception": "{}".format(e),
                                  "camd_traceback": traceback.format_exc()})
                calc.update({
                    "status": "FAILED",
                    "error": error_doc,
                    "result": None
                })
        elif aws_status == "FAILED":
            error_doc = {"aws_fail": result['jobs'][0]['attempts'][-1]['statusReason']}
            calc.update({"status": "FAILED",
                         "error": error_doc
                         })
        else:
            calc.update({"status": aws_status})
    return calc_status



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
    os.system("mpirun -n 1 vasp_std")
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
    os.system("mpirun -n 1 vasp_std")
    os.chdir('..')


if __name__ == '__main__':
    run_oqmd_calculation("POSCAR")
"""


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
    raise NotImplementedError("Northwestern interface not yet implemented")