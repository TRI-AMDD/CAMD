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
import pandas as pd
import datetime


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

    @property
    def state(self):
        if self.get_state() and self.job_status:
            return "completed"
        elif sum([doc['status'] in ['SUCCEEDED', 'FAILED']
                    for doc in self.job_status.values()]) > 0:
            return "active"
        else:
            return "unstarted"

    @property
    def state_detailed(self):
        return NotImplementedError("Not implemented")

    def get_state(self):
        self.job_status = check_dft_calcs(self.job_status)
        return all([doc['status'] in ['SUCCEEDED', 'FAILED']
                    for doc in self.job_status.values()])

    def get_results(self, indices, populate_candidate_data=True):
        # This gets the formation energies.
        if not self.get_state():
            warnings.warn("Some calculations did not finish.")
        results_dict = {}
        for structure_id, calc in self.job_status.items():
            if calc['status'] == "SUCCEEDED":
                delta_e = get_qmpy_formation_energy(calc['result']['output']['final_energy_per_atom'],
                                                    calc['result']['pretty_formula'], 1)

                results_dict[structure_id] = delta_e

        if populate_candidate_data:
            candidate_data = self.get_parameter("candidate_data")
            _df = candidate_data.loc[results_dict.keys()]
            _df['delta_e'] = pd.Series(results_dict)
            _df = _df.reindex(sorted(_df.columns), axis=1)
            return _df
        else:
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

    def monitor(self):
        """
        Returns:
            (dict): calculation status, including results
        """
        finished = self.get_state()
        with ScratchDir("."):
            while not finished:
                time.sleep(self.poll_time)
                finished = self.get_state()
                for doc in self.job_status.values():
                    doc["elapsed_time"] = time.time() - doc["start_time"]
                status_string = "\n".join(["{}: {} {}".format(key, value["status"],
                                                              datetime.timedelta(0,value["elapsed_time"]))
                                           for key, value in self.job_status.items()])
                print("Calc status:\n{}".format(status_string))
                print("Timeout is set as {}.".format(datetime.timedelta(0, self.timeout)))

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
    def from_job_status(cls, params, job_status):
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
            print("Submitting job: {}".format(structure_id))
            try:
                calc = subprocess.check_output(["trisub", "-q", "oqmd_test_queue", "-r", "16000", "-c", "16"])
            except subprocess.CalledProcessError as e:
                print(e.output)

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


def get_qmpy_formation_energy(total_e, formula, n_atoms):
    mus = {u'Ac': -4.1060035325, u'Ag': -2.8217729525, u'Al': -3.74573946, u'Ar': -0.00636995, u'As': -4.651918435,
           u'Au': -3.26680174, u'B': -6.67796758, u'Ba': -1.92352708, u'Be': -3.75520865, u'Bi': -4.038931855,
           u'Br': -1.31759562258416, u'C': -9.2170759925, u'Ca': -1.977817, u'Cd': -0.90043514, u'Ce': -4.7771708225,
           u'Cl': -1.47561368438088, u'Co': -7.089565, u'Cr': -9.50844998, u'Cs': -0.85462775, u'Cu': -3.7159594,
           u'Dy': -4.60150328333333, u'Er': -4.56334055, u'Eu': -1.8875732, u'F': -1.45692429086889, u'Fe': -8.3078978,
           u'Ga': -3.031846515, u'Gd': -4.6550712925, u'Ge': -4.623692585, u'H': -3.38063384781582, u'He': -0.004303435,
           u'Hf': -9.955368785, u'Hg': -0.358963825033731, u'Ho': -4.57679364666667, u'I': -1.35196205757168,
           u'In': -2.71993876, u'Ir': -8.8549203, u'K': -1.096699335, u'Kr': -0.004058825, u'La': -4.93543556,
           u'Li': -1.89660627, u'Lu': -4.524181525, u'Mg': -1.54251595083333, u'Mn': -9.0269032462069,
           u'Mo': -10.8480839, u'N': -8.11974103465649, u'Na': -1.19920373914835, u'Nb': -10.09391206,
           u'Nd': -4.762916335, u'Ne': -0.02931791, u'Ni': -5.56661952, u'Np': -12.94027372125, u'O': -4.52329546412125,
           u'Os': -11.22597601, u'P': -5.15856496104006, u'Pa': -9.49577589, u'Pb': -3.70396484, u'Pd': -5.17671826,
           u'Pm': -4.7452352875, u'Pr': -4.7748066125, u'Pt': -6.05575959, u'Pu': -14.29838348, u'Rb': -0.9630733,
           u'Re': -12.422818875, u'Rh': -7.26940476, u'Ru': -9.2019888, u'S': -3.83888286598664, u'Sb': -4.117563025,
           u'Sc': -6.328367185, u'Se': -3.48117276, u'Si': -5.424892535, u'Sm': -4.7147675825, u'Sn': -3.9140929231488,
           u'Sr': -1.6829138, u'Ta': -11.85252937, u'Tb': -5.28775675533333, u'Tc': -10.360747885,
           u'Te': -3.14184237666667, u'Th': -7.41301719, u'Ti': -7.69805778621374, u'Tl': -2.359420025,
           u'Tm': -4.47502416, u'U': -11.292348705, u'V': -8.94097896, u'W': -12.96020695, u'Xe': 0.00306349,
           u'Y': -6.464420635, u'Yb': -1.51277545, u'Zn': -1.2660268, u'Zr': -8.54717235}
    hubbard_mus = {u'Co': 2.0736240219357, u'Cr': 2.79591214925926, u'Cu': 1.457571831687, u'Fe': 2.24490453841424,
                   u'Mn': 2.08652912841877, u'Ni': 2.56766185643768, u'Np': 2.77764768949249, u'Pu': 2.2108747749433,
                   u'Th': 1.06653674624248, u'U': 2.57513786752409, u'V': 2.67812162528461}

    from pymatgen import Composition
    c = Composition(formula).fractional_composition
    e = total_e / n_atoms
    for k,v in c.as_dict().items():
        e -= mus[k]*v
        if (k in hubbard_mus) and ('O' in c):
            e-=hubbard_mus[k]*v
    return e