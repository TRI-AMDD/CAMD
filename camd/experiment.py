# Copyright Toyota Research Institute 2019

import os
import uuid
import shutil
from monty.os import cd
from pymatgen.io.vasp.outputs import Vasprun
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


OQMD_MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.py")


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
    uuid_string = uuid.uuid4()
    parent_dir = os.path.join(tri_path, "model", "oqmdvasp", "2",
                              "u", "camd", "run_{}".format(uuid_string))
    if any(['_' in key for key in structure_dict.keys()]):
        raise ValueError("Structure keys cannot contain underscores for "
                         "mc1 compatibility")

    calc_status = {}
    for structure_id, structure in structure_dict.items():
        calc_path = os.path.join(parent_dir, structure_id)
        os.makedirs(calc_path)
        with cd(calc_path):
            # Write input cif file and python model file
            structure.to(filename="input.cif")
            shutil.copy(OQMD_MODEL_FILE, "model.py")

            # Submit to mc1
            # TODO: ensure this is checked for failure to submit
            os.system("trisub")

            # Add status to status doc
            calc_status[structure_id] = {
                "path": os.getcwd(), "status": "pending"}
    return calc_status


def check_dft_calcs(calc_status):
    """

    Args:
        calc_status (dict):

    Returns:
        updated calc_status dictionary

    """
    for structure_id, status_doc in calc_status.items():
        if status_doc['status'] in ['completed', 'failed']:
            continue
        path = status_doc['path']
        print("Checking status of {}: {}".format(path, structure_id))
        os.chdir(path)
        os.system('trisync')
        os.chdir('simulation')
        # TODO: look into querying AWS batch system, rather than filesystem
        if os.path.isfile('completed'):
            try:
                os.chdir('static')
                vr = Vasprun('.')
                status_doc.update({
                    "state": "completed",
                    "error": None,
                    "result": vr.as_dict()
                })
            except Exception as e:
                error_doc = {}
                with open('err') as errfile:
                    error_doc.update({"trisub_stderr": errfile.read()})
                error_doc.update({"camd_exception": "{}".format(e),
                                  "camd_traceback": traceback.format_exc(e)})
                status_doc.update({
                    "state": "failed",
                    "error": error_doc,
                    "result": None
                })
        else:
            pass
            # TODO: could put some checkpoints here for viz, etc.
    return calc_status
