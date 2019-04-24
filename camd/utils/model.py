import os

import qmpy
from qmpy.materials.structure import Structure
from qmpy.analysis.vasp.calculation import Calculation
from qmpy import io


# TODO: definitely move this somewhere else
def run_oqmd_calculation(cif_filename):
    starting_structure = io.cif.read(cif_filename)

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
    run_oqmd_calculation("input.cif")
