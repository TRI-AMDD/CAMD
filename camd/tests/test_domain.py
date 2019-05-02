import unittest
from camd.domain import create_structure_dataframe


class DomainTest(unittest.TestCase):
    def test_create_structure_dataframe(self):
        structure_df = create_structure_dataframe(["Al2O3", "AlO4"])


if __name__ == '__main__':
    unittest.main()
