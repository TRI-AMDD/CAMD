"""
Preliminary module for determining search spaces
"""
import os
import pandas as pd
import abc

from camd import S3_CACHE
from camd.utils.s3 import cache_s3_objs
from protosearch.build_bulk.oqmd_interface import OqmdInterface

# Just an initial idea, will need some fleshing out
class Domain(abc.ABC):
    @abc.abstractmethod
    def sample(self, num_samples):
        """
        Abstract method for sampling from a domain

        Args:
            num_samples:

        Returns:

        """
        pass


# TODO: sample from a chemsys, as opposed to formulas
def create_structure_dataframe(formulas, db_interface=None):
    """
    Function to create a dataframe of structures corresponding
    to formulas from OQMD prototypes

    Args:
        formulas ([str]): list of chemical formulas from which
            to generate candidate structures
        db_interface (DbInterface): interface to OQMD database
            by default uses the one stored in s3

    Returns:

    """
    if db_interface is None:
        obj = "camd/shared-data/protosearch-data/materials-db/oqmd/oqmd_ver3.db"
        cache_s3_objs([obj])
        oqmd_db_path = os.path.join(S3_CACHE, obj)
        db_interface = OqmdInterface(oqmd_db_path)
    dataframes = [db_interface.create_proto_data_set(formula)
                  for formula in formulas]
    return pd.concat(dataframes)
