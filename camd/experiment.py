# Copyright Toyota Research Institute 2019

import os
import uuid
import json
import re
from monty.os import cd
import shlex
from pymatgen.io.vasp.outputs import Vasprun
import subprocess
import traceback

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
    raise NotImplementedError("Northwestern interface not yet implemented")


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
    parent_dir = os.path.join(tri_path, "model", "oqmdvasp", "3",
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
            print("Submitting job")
            calc = subprocess.check_output(["trisub", "-q", "oqmd_test_queue"])
            calc = calc.decode('utf-8')
            calc = re.findall("({.+})", calc, re.DOTALL)[0]
            calc = json.loads(calc)
            calc.update({"path": os.getcwd(),
                         "status": "SUBMITTED"})
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
        # TODO: probably could use botocore for this
        aws_cmd = "aws batch describe-jobs --jobs {}".format(calc['jobId'])
        result = subprocess.check_output(shlex.split(aws_cmd))
        result = json.loads(result)
        aws_status = result["jobs"][0]["status"]
        # calc.update({"status": status})
        if aws_status is "SUCCEEDED":
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
        elif aws_status is "FAILED":
            error_doc = {"aws_fail": result['attempts'][-1]['statusReason']}
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
