#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module consolidates s3-based dataset loading
for CAMD, mostly in the form of dataframes
"""

import os

import boto3
import botocore
import requests
import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from monty.os import makedirs_p
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty,
)
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    StructuralHeterogeneity,
    ChemicalOrdering,
    StructureComposition,
    MaximumPackingEfficiency,
)
from camd import CAMD_CACHE, tqdm


def load_dataframe(dataset_name):
    """
    Helper function to load a dataframe from the utilities below

    Supported datasets include:
        * oqmd_1.2_voronoi_magpie_fingerprints
        * oqmd1.2_exp_based_entries_featurized_v2,

    Args:
        dataset_name (str): dataset name to load

    Returns:
        (DataFrame): dataframe corresponding to specified dataset

    """
    filename = '{}.pickle'.format(dataset_name)
    cache_matrio_data(filename)
    return pd.read_pickle(os.path.join(CAMD_CACHE, filename))


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


def filter_dataframe_by_composition(df, composition, formula_column="Composition"):
    """
    Filters dataframe by composition, i. e. finds all
    rows in dataframe where the Composition contains a
    subset of input composition

    Args:
        df (DataFrame): dataframe
        composition (Composition or str): composition
            or formula by which to filter
        formula_column (str): formula column name, defaults
            to "Composition"

    Returns:
        (DataFrame): dataframe where every composition is sampled such
            that its composition is a subset of the input element set

    """
    # Get elements in formula, composition, then filter
    chemsys = set(Composition(composition).keys())
    all_comps = df[formula_column].apply(Composition)
    indices_to_include = [ind for ind, comp in all_comps.items()
                          if comp.keys() <= chemsys]
    return df.loc[indices_to_include]


def get_oqmd_data_by_chemsys(chemsys, drop_duplicates=True):
    """
    Utility function for loading a chemsys from cached
    OQMD data, used primarily for campaign simulations

    Args:
        chemsys (str): formula or hyphen separated list
            of elements, e. g. FeO2, Fe-Cl, etc.
        drop_duplicates (bool): whether to drop duplicate
            values from dataframe

    Returns:
        (DataFrame): dataframe corresponding to oqmd
            data from chemsys

    """
    all_data = load_dataframe("oqmd_1.2_voronoi_magpie_fingerprints")
    dataset = filter_dataframe_by_composition(
        all_data, chemsys.replace('-', '')
    )
    if drop_duplicates:
        dataset = dataset.drop_duplicates(keep='first')
    return dataset

def get_mp_ele_ref(element):
    """
    Helper function to get the MP compatible elemental references
    to calculate the formation energy
    Args:
        element (str): the element
    Returns:
        the minimum energy of the unary phase of the element from MP
    """
    with MPREster() as mpr:
        energies = [e.energy_per_atom for e in mpr.get_entries_in_chemsys([element])]
    return min(energies)

MP_REFERENCES = {"H": -3.39271585, "He": -0.00902216, "Li": -1.9089228533333333, "Be": -3.739412865,
                "B": -6.679390641666667, "C": -9.22676982, "N": -8.336493965, "O": -4.94795546875, "F": -1.9114557725,
                "Ne": -0.02582742, "Na": -1.312223005, "Mg": -1.5968959166666667, "Al": -3.74557583, "Si": -5.423390655,
                "P": -5.413285861428571, "S": -4.136449866875, "Cl": -1.8485262325, "Ar": -0.06880822,
                "K": -1.110398947, "Ca": -1.999462735, "Sc": -6.332469105, "Ti": -7.895052840000001, "V": -9.08235617,
                "Cr": -9.65304747, "Mn": -9.161706470344827, "Fe": -8.46929704, "Co": -7.108317795, "Ni": -5.7798218,
                "Cu": -4.09920667, "Zn": -1.259460605, "Ga": -3.02808992, "Ge": -4.61751163, "As": -4.65850806,
                "Se": -3.49591147765625, "Br": -1.63692654, "Kr": -0.05671467, "Rb": -0.9805340725,
                "Sr": -1.6894812533333334, "Y": -6.466074656666667, "Zr": -8.54770063, "Nb": -10.10130504,
                "Mo": -10.84563514, "Tc": -10.36061991, "Ru": -9.27438911, "Rh": -7.33850956, "Pd": -5.17648694,
                "Ag": -2.8325290566666665, "Cd": -0.90620278, "In": -2.75168373, "Sn": -3.99229498, "Sb": -4.128999585,
                "Te": -3.14330093, "I": -1.524009065, "Xe": -0.03617417, "Cs": -0.8954023720689656, "Ba": -1.91897083,
                "La": -4.936007105, "Ce": -5.933089155, "Pr": -4.780899145, "Nd": -4.76814321, "Pm": -4.7505423225,
                "Sm": -4.717682476666667, "Eu": -10.292043475, "Gd": -14.07612224, "Tb": -4.6343661, "Dy": -4.60678684,
                "Ho": -4.58240887, "Er": -4.5674248, "Tm": -4.475058396666666, "Yb": -1.5395952733333333,
                "Lu": -4.52095052, "Hf": -9.95718477, "Ta": -11.85777728, "W": -12.95812647, "Re": -12.444527185,
                "Os": -11.22733445, "Ir": -8.83843042, "Pt": -6.07090771, "Au": -3.27388154, "Hg": -0.30362902,
                "Tl": -2.36165292, "Pb": -3.71258955, "Bi": -3.88641638, "Ac": -4.1211750075, "Th": -7.41385825,
                "Pa": -9.51466466, "U": -11.29141001, "Np": -12.94777968125, "Pu": -14.26782579}

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

ELEMENTS = [
    "Ru",
    "Re",
    "Rb",
    "Rh",
    "Be",
    "Ba",
    "Bi",
    "Br",
    "H",
    "P",
    "Os",
    "Ge",
    "Gd",
    "Ga",
    "Pr",
    "Pt",
    "Pu",
    "C",
    "Pb",
    "Pa",
    "Pd",
    "Xe",
    "Pm",
    "Ho",
    "Hf",
    "Hg",
    "He",
    "Mg",
    "K",
    "Mn",
    "O",
    "S",
    "W",
    "Zn",
    "Eu",
    "Zr",
    "Er",
    "Ni",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Np",
    "Fe",
    "B",
    "F",
    "Sr",
    "N",
    "Kr",
    "Si",
    "Sn",
    "Sm",
    "V",
    "Sc",
    "Sb",
    "Se",
    "Co",
    "Cl",
    "Ca",
    "Ce",
    "Cd",
    "Tm",
    "Cs",
    "Cr",
    "Cu",
    "La",
    "Li",
    "Tl",
    "Lu",
    "Th",
    "Ti",
    "Te",
    "Tb",
    "Tc",
    "Ta",
    "Yb",
    "Dy",
    "I",
    "U",
    "Y",
    "Ac",
    "Ag",
    "Ir",
    "Al",
    "As",
    "Ar",
    "Au",
    "In",
    "Mo",
]


def cache_download(url, path):
    """
    Quick helper function to cache a generic download from a url
    in the CAMD local data directory

    Args:
        url (str): url for download
        path (str): path for download, is appended to the
            CAMD_CACHE location

    Returns:
        (None)
    """
    # Prep cache path and make necessary dirs
    cache_path = os.path.join(CAMD_CACHE, path)

    # Download and write file
    if not os.path.isfile(cache_path):
        makedirs_p(os.path.split(cache_path)[0])
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(cache_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)


# Mapping of matrio data hashes to cache paths
MATRIO_DATA_KEYS = {
    "oqmd1.2_exp_based_entries_featurized_v2.pickle": "5e3b0e9bc91e209071f33ce8",
    "oqmd_1.2_voronoi_magpie_fingerprints.pickle": "5e39ce2cd9f13e075b7dfaaf",
    "oqmd_ver3.db": "5e39ce96d9f13e075b7dfab3",
    "oqmd1.2_exp_based_entries_structures.json": "5e45befef23b399192b35242"
}


def get_chemsys(formula_or_structure, seperator='-'):
    """
    Gets a sorted, character-delimited set of elements, e.g.
    Fe-Ni-O or O-Ti

    Args:
        formula_or_structure (str, Structure): formula or structure
            for which to get chemical system
        separator (str): separator for the chemsys elements

    Returns:
        (str): separated

    """
    if isinstance(formula_or_structure, Structure):
        formula = formula_or_structure.composition.reduced_formula
    else:
        formula = formula_or_structure
    elements = [str(el) for el in Composition(formula)]
    return seperator.join(sorted(elements))


def cache_matrio_data(filename):
    """
    Helper function to cache data hosted on data.matr.io

    Args:
        filename (filename): filename to fetch from data.matr.io

    Returns:
        None

    """
    prefix = "https://data.matr.io/3/api/v1/file"
    key = MATRIO_DATA_KEYS[filename]
    if not os.path.isfile(filename):
        cache_download("{}/{}/download".format(prefix, key), filename)


def partition_intercomp(dataframe, n_elements=None):
    """
    Utility function to partition a dataframe into
    data in the interior of the phase diagram

    Args:
        dataframe (DataFrame): dataframe to be partitioned
        n_elements (int): number of elements by which to
            partition the dataframe, defaults to n-1, where n
            is the total number of elements in the dataframe

    Returns:
        (DataFrame): data which contain elements in the
            interior of the phase space according to the
            n_element threshold
        (DataFrame): data which are below the n_element
            threshold
    """
    all_n_elts = [len(Composition(formula))
                  for formula in dataframe['Composition']]
    if n_elements is None:
        n_elements = max(all_n_elts) - 1

    mask = np.array(all_n_elts) > n_elements
    return dataframe[mask], dataframe[~mask]


def s3_sync(s3_bucket, s3_prefix, sync_path="."):
    """
    Syncs a given path to an s3 prefix

    Args:
        s3_bucket (str): bucket name
        s3_prefix (str): s3 prefix to sync to
        sync_path (str, Path): path to sync to bucket:prefix

    Returns:
        (None)

    """
    # Get bucket
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(s3_bucket)

    # Walk paths and subdirectories, uploading files
    for path, subdirs, files in os.walk(sync_path):
        # Get relative path prefix
        relpath = os.path.relpath(path, sync_path)
        if not relpath.startswith('.'):
            prefix = os.path.join(s3_prefix, relpath)
        else:
            prefix = s3_prefix

        for file in files:
            file_key = os.path.join(prefix, file)
            bucket.upload_file(os.path.join(path, file), file_key)


def s3_key_exists(key, bucket):
    """
    Quick utility to determine whether key exists in bucket

    Args:
        key (str): key to check
        bucket (str): bucket to check in

    Returns:
        (bool): whether the key was found in the bucket

    """
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            return False
        else:
            # Something else has gone wrong.
            raise e
    else:
        # The object does exist.
        return True


def download_s3_file(key, bucket, output_filename):
    """
    Quick utility to download s3 file

    Args:
        key (str): key to download
        bucket (str): bucket from which to download
        output_filename (str): output filename for object

    Returns:
        (bool): whether the key was found in the bucket

    """
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, output_filename)
    return True


def upload_s3_file(key, bucket, filename):
    """
    Quick utility to upload s3 file

    Args:
        key (str): key to download
        bucket (str): bucket from which to download
        filename (str): output filename for object

    Returns:
        (bool): whether the key was found in the bucket

    """
    s3_client = boto3.client('s3')
    s3_client.upload_file(filename, bucket, key)
    return True


def get_common_prefixes(bucket, prefix):
    """
    Helper function to get common "subfolders" of folders
    in S3

    Args:
        bucket (str): bucket name
        prefix (str): prefix for which to list common prefixes

    Returns:

    """
    if not prefix.endswith('/'):
        prefix += "/"
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects')
    result = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix)
    return [common_prefix['Prefix'].split('/')[-2]
            for common_prefix in result.search("CommonPrefixes")
            if common_prefix]


def get_default_featurizer():
    """
    Utility function to get CAMD's default featurizer from Ward et al. (2017)

    Returns:
        (Featurizer): default CAMD featurizer

    """
    return MultipleFeaturizer(
        [
            SiteStatsFingerprint.from_preset(
                "CoordinationNumber_ward-prb-2017"
            ),
            StructuralHeterogeneity(),
            ChemicalOrdering(),
            MaximumPackingEfficiency(),
            SiteStatsFingerprint.from_preset(
                "LocalPropertyDifference_ward-prb-2017"
            ),
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=["frac"])),
            StructureComposition(IonProperty(fast=True)),
        ]
    )
