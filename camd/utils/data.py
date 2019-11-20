#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module consolidates s3-based dataset loading
for CAMD, mostly in the form of dataframes
"""

import os
import pandas as pd
from camd import S3_CACHE
from camd.utils.s3 import cache_s3_objs

# TODO-PUBLIC: this will need to be made general for the
#   release of the code, mostly by making the s3 bucket public


def load_dataframe(dataset_name):
    """
    Supported datasets include:
        * oqmd_1.2_voronoi_magpie_fingerprints

    Args:
        dataset_name (str): dataset name to load

    Returns:
        (DataFrame): dataframe corresponding to specified dataset

    """
    key = 'camd/shared-data/{}.pickle'.format(dataset_name)
    cache_s3_objs([key])
    return pd.read_pickle(os.path.join(S3_CACHE, key))


def load_default_atf_data():
    """
    Convenience function to load the default test ATF data
    which is a dataframe of OQMD data of binary compounds with
    magpie and voronoi analysis features

    Returns:
        (DataFrame): dataframe of OQMD data and the corresponding
            features

    """
    df = load_dataframe("oqmd_1.2_voronoi_magpie_fingerprints")
    return df[df['N_species'] == 2].sample(frac=0.2)


QMPY_REFERENCES = {
    u'Ac': -4.1060035325,
    u'Ag': -2.8217729525,
    u'Al': -3.74573946,
    u'Ar': -0.00636995,
    u'As': -4.651918435,
    u'Au': -3.26680174,
    u'B': -6.67796758,
    u'Ba': -1.92352708,
    u'Be': -3.75520865,
    u'Bi': -4.038931855,
    u'Br': -1.31759562258416,
    u'C': -9.2170759925,
    u'Ca': -1.977817,
    u'Cd': -0.90043514,
    u'Ce': -4.7771708225,
    u'Cl': -1.47561368438088,
    u'Co': -7.089565,
    u'Cr': -9.50844998,
    u'Cs': -0.85462775,
    u'Cu': -3.7159594,
    u'Dy': -4.60150328333333,
    u'Er': -4.56334055,
    u'Eu': -1.8875732,
    u'F': -1.45692429086889,
    u'Fe': -8.3078978,
    u'Ga': -3.031846515,
    u'Gd': -4.6550712925,
    u'Ge': -4.623692585,
    u'H': -3.38063384781582,
    u'He': -0.004303435,
    u'Hf': -9.955368785,
    u'Hg': -0.358963825033731,
    u'Ho': -4.57679364666667,
    u'I': -1.35196205757168,
    u'In': -2.71993876,
    u'Ir': -8.8549203,
    u'K': -1.096699335,
    u'Kr': -0.004058825,
    u'La': -4.93543556,
    u'Li': -1.89660627,
    u'Lu': -4.524181525,
    u'Mg': -1.54251595083333,
    u'Mn': -9.0269032462069,
    u'Mo': -10.8480839,
    u'N': -8.11974103465649,
    u'Na': -1.19920373914835,
    u'Nb': -10.09391206,
    u'Nd': -4.762916335,
    u'Ne': -0.02931791,
    u'Ni': -5.56661952,
    u'Np': -12.94027372125,
    u'O': -4.52329546412125,
    u'Os': -11.22597601,
    u'P': -5.15856496104006,
    u'Pa': -9.49577589,
    u'Pb': -3.70396484,
    u'Pd': -5.17671826,
    u'Pm': -4.7452352875,
    u'Pr': -4.7748066125,
    u'Pt': -6.05575959,
    u'Pu': -14.29838348,
    u'Rb': -0.9630733,
    u'Re': -12.422818875,
    u'Rh': -7.26940476,
    u'Ru': -9.2019888,
    u'S': -3.83888286598664,
    u'Sb': -4.117563025,
    u'Sc': -6.328367185,
    u'Se': -3.48117276,
    u'Si': -5.424892535,
    u'Sm': -4.7147675825,
    u'Sn': -3.9140929231488,
    u'Sr': -1.6829138,
    u'Ta': -11.85252937,
    u'Tb': -5.28775675533333,
    u'Tc': -10.360747885,
    u'Te': -3.14184237666667,
    u'Th': -7.41301719,
    u'Ti': -7.69805778621374,
    u'Tl': -2.359420025,
    u'Tm': -4.47502416,
    u'U': -11.292348705,
    u'V': -8.94097896,
    u'W': -12.96020695,
    u'Xe': 0.00306349,
    u'Y': -6.464420635,
    u'Yb': -1.51277545,
    u'Zn': -1.2660268,
    u'Zr': -8.54717235}

QMPY_REFERENCES_HUBBARD = {
    u'Co': 2.0736240219357,
    u'Cr': 2.79591214925926,
    u'Cu': 1.457571831687,
    u'Fe': 2.24490453841424,
    u'Mn': 2.08652912841877,
    u'Ni': 2.56766185643768,
    u'Np': 2.77764768949249,
    u'Pu': 2.2108747749433,
    u'Th': 1.06653674624248,
    u'U': 2.57513786752409,
    u'V': 2.67812162528461
}

