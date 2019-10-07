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

