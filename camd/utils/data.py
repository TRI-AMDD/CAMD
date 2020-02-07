#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module consolidates s3-based dataset loading
for CAMD, mostly in the form of dataframes
"""

import os
import requests
import pandas as pd
from monty.os import makedirs_p
from camd import CAMD_CACHE, tqdm

# TODO-PUBLIC: this will need to be made general for the
#   release of the code, mostly by making the s3 bucket public


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
        r = requests.get(url)
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
    "oqmd_ver3.db": "5e39ce96d9f13e075b7dfab3"
}


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
