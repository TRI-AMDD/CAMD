"""
This module contains basic functionality related to datasets. Classes in this
module should support making it easy for users to retrieve and use datasets by
name. Code in this module should not be specific to any particular dataset.
"""

from os.path import join

import pandas as pd
from camd import CAMD_ROOT  # , CAMD_S3_BUCKET
from monty.json import MSONable


class Dataset(MSONable):
    def __init__(
        self,
        name: str,
        data_dir: str = None,
        # s3_bucket: str = CAMD_S3_BUCKET,
        # s3_prefix: str = "datasets",
        # s3_profile: str = None,
    ) -> None:
        self.name = name
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = join(CAMD_ROOT, "data")

        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        fname = f"{self.name}.csv"
        file_path = join(self.data_dir, fname)
        return pd.read_csv(file_path)
