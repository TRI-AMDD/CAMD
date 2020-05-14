import unittest
import os
from pymatgen import Structure

from camd.domain import StructureDomain, get_structures_from_protosearch, heuristic_setup


CAMD_LONG_TESTS = os.environ.get("CAMD_LONG_TESTS", False)
SKIP_MSG = "Long tests disabled, set CAMD_LONG_TESTS to run long tests"


# TODO: add lightweight version of these?
class DomainTest(unittest.TestCase):
    def test_get_structures_from_protosearch(self):
        structure_df = get_structures_from_protosearch(["V3O7"], source='icsd')

        self.assertEqual(structure_df.shape, (20, 11))

    @unittest.skipUnless(CAMD_LONG_TESTS, SKIP_MSG)
    def test_StructureDomain(self):
        sd = StructureDomain.from_bounds(['Ir', 'O'], charge_balanced=False)

        self.assertEqual(sd.bounds, {'O', 'Ir'})
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
        self.assertEqual(type(sd.hypo_structures['pmg_structures'][3]), Structure)

        sd.featurize_structures()
        self.assertEqual(sd.features.shape, (43, 275))

    @unittest.skipUnless(CAMD_LONG_TESTS, SKIP_MSG)
    def test_heuristic_setup(self):
        element_list = ['Al', 'Fe']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertEqual(g_max, 5)
        self.assertFalse(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 11)
        self.assertIn('Al3Fe4', sd.formulas)

        element_list = ['Al', 'Fe', 'Ti']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertEqual(g_max, 5)
        self.assertFalse(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 55)
        self.assertIn('Al3Fe2Ti2', sd.formulas)

        element_list = ['Al', 'Fe', 'Ti', 'Mn']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertEqual(g_max, 4)
        self.assertFalse(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 79)
        self.assertIn('Al2Fe2Ti2Mn3', sd.formulas)

        element_list = ['Al', 'O']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertTrue(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 1)
        self.assertIn('Al2O3', sd.formulas)

        element_list = ['Fe', 'O']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertTrue(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 7)
        self.assertIn('Fe3O4', sd.formulas)

        element_list = ['Al', 'Fe', 'O']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertTrue(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 15)
        self.assertIn('Al2Fe2O5', sd.formulas)

        element_list = ['Al', 'Fe', 'Ti', 'O']
        g_max, charge_balanced = heuristic_setup(element_list)
        self.assertTrue(charge_balanced)
        sd = StructureDomain.from_bounds(element_list, charge_balanced=charge_balanced,
                                             **{'grid': range(1, g_max)})
        self.assertEqual(len(sd.formulas), 27)
        self.assertIn('Al2Fe1Ti3O7', sd.formulas)


if __name__ == '__main__':
    unittest.main()
