"""

unit tests for feature computation and provision

"""

import unittest

from camd.database.schema import Material
from camd.model.feature.provide import FeatureComputer
from camd.utils.postgres import sqlalchemy_session
from camd.database.access import CamdSchemaSession
from camd.utils.postgres import database_available

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, \
    ValenceOrbital, IonProperty
from matminer.featurizers.structure import SiteStatsFingerprint, \
    StructuralHeterogeneity, ChemicalOrdering, StructureComposition
from matminer.featurizers.structure import MaximumPackingEfficiency


ENVIRONMENT = 'local'


@unittest.skipUnless(database_available(ENVIRONMENT),
                     ENVIRONMENT + ' database not available')
class TestFeatureComputation(unittest.TestCase):

    def setUp(self):

        # populate some materials (if they don't yet exist in db)
        bucket = 'oqmd-chargedensity'
        s3_key_list = ['OQMD_CHGCARs/1000000_POSCAR',
                            'OQMD_CHGCARs/1000001_POSCAR']
        session = sqlalchemy_session(ENVIRONMENT)
        for key in s3_key_list:
            material = Material.from_poscar_s3(bucket, prefix=key,
                                               internal_reference=key,
                                               dft_computed=True)
            try:
                session.add(material)
                session.commit()
            except Exception as e:
                print(e)
                session.rollback()

    def tearDown(self):
        pass

    def _actual_features(self, structure):
        featurizer = MultipleFeaturizer([
            SiteStatsFingerprint.from_preset(
                "CoordinationNumber_ward-prb-2017"),
            StructuralHeterogeneity(),
            ChemicalOrdering(),
            MaximumPackingEfficiency(),
            SiteStatsFingerprint.from_preset(
                "LocalPropertyDifference_ward-prb-2017"),
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=['frac'])),
            StructureComposition(IonProperty(fast=True))
        ])

        features = featurizer.featurize(structure)
        self.assertTrue(isinstance(features[270], bool))
        features[270] = float(features[270])
        return features, featurizer.feature_labels()

    def test_compute_all_features(self):

        css = CamdSchemaSession(ENVIRONMENT)


        # material 1
        mat1 = css.query_material_by_internal_reference(
            'OQMD_CHGCARs/1000000_POSCAR')
        structure1 = mat1.structure()

        fc = FeatureComputer()
        featurizations, feature_labels, _ = fc.compute_all_features(mat1)
        features_actual, labels_actual = self._actual_features(structure1)

        self.assertEqual(len(featurizations), len(features_actual))
        self.assertEqual(len(feature_labels), len(labels_actual))
        self.assertEqual(len(featurizations), len(feature_labels))

        for i in range(len(featurizations)):
            self.assertEqual(features_actual[i], featurizations[i])
            self.assertEqual(labels_actual[i], feature_labels[i])

        # material 2
        mat2 = css.query_material_by_internal_reference(
            'OQMD_CHGCARs/1000001_POSCAR')
        structure2 = mat2.structure()

        fc = FeatureComputer()
        featurizations, feature_labels, _ = fc.compute_all_features(mat2)
        features_actual, labels_actual = self._actual_features(structure2)

        self.assertEqual(len(featurizations), len(features_actual))
        self.assertEqual(len(feature_labels), len(labels_actual))
        self.assertEqual(len(featurizations), len(feature_labels))

        for i in range(len(featurizations)):
            self.assertEqual(features_actual[i], featurizations[i])
            self.assertEqual(labels_actual[i], feature_labels[i])