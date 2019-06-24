import unittest
from camd.domain import StructureDomain, get_structures_from_protosearch
from camd.utils.s3 import s3_connection_broken
from pymatgen import Structure


if not s3_connection_broken("matr.io", "camd/shared-data"):
    S3_CONNECTION = True
else:
    S3_CONNECTION = False


# TODO:
# - Add StructureDomain tests
@unittest.skipUnless(S3_CONNECTION, "S3 connection broken")
class DomainTest(unittest.TestCase):

    def test_get_structures_from_protosearch(self):
        structure_df = get_structures_from_protosearch(["V3O7"], source='icsd')

        self.assertEqual(structure_df.shape, (20, 4))
        # self.assertIn("A3B7_2_b2_a2b4_146_O_V", list(structure_df.index))

    def test_StructureDomain(self):
        sd = StructureDomain.from_bounds(['Ir', 'O'], charge_balanced=False)

        self.assertEqual(sd.bounds, {'O', 'Ir'} )
        self.assertTrue(len(sd.formulas), 35)
        self.assertIn('Ir7O4', sd.formulas)

        sd = StructureDomain.from_bounds(['Ir', 'O'], charge_balanced=True)
        self.assertTrue(len(sd.formulas), 6)
        self.assertNotIn('Ir7O4', sd.formulas)
        self.assertNotIn('Ir1O3', sd.formulas)

        kwargs = {"oxi_states_extend": {"Ir": [6]}}
        sd = StructureDomain.from_bounds(['Ir', 'O'], **kwargs)
        self.assertIn('Ir1O3', sd.formulas)

        kwargs = {"oxi_states_extend": {"Ir": [6]},  "grid": list(range(1, 4))}
        sd = StructureDomain.from_bounds(['Ir', 'O'], **kwargs)
        self.assertTrue(len(sd.formulas), 3)
        sd = StructureDomain.from_bounds(['Li', 'Ir', 'O'], **kwargs)
        self.assertIn('Li1Ir1O2', sd.formulas)

        kwargs = {"grid": list(range(1, 9))}
        sd = StructureDomain.from_bounds(['Y', 'Ba', 'Cu', 'O'], **kwargs)
        self.assertEqual(len(sd.formulas), 112)

        kwargs = {"grid": list(range(1, 4))}
        sd = StructureDomain.from_bounds(['Y', 'Ba', 'Cu', 'O'], **kwargs)
        self.assertEqual(len(sd.formulas), 1)
        sd.get_structures()
        self.assertEqual(len(sd.hypo_structures), 43)
        self.assertEqual(type(sd.hypo_structures['pmg_structures'][3]),Structure)

        sd.featurize_structures()
        self.assertEqual(sd.features.shape, (43, 275))
        # self.assertAlmostEqual(sd.features.loc['ABCD3_2_g_i_i_ij_12_Ba_Cu_O_Y']['range CN_VoronoiNN'],
        #                        6.6949386, places=6)


if __name__ == '__main__':
    unittest.main()
