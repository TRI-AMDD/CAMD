"""

unit tests for feature computation and provision

"""

import logging, sys
import unittest

from camd.database.schema import Material
from camd.model.feature.provide import FeatureProvider
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

# logger
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)


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
                logging.debug('Material already in database.')
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

    def test_provide_block_featurization(self):
        css = CamdSchemaSession(ENVIRONMENT)

        # actual
        internal_references = ['OQMD_CHGCARs/1000000_POSCAR',
                               'OQMD_CHGCARs/1000001_POSCAR']
        material_ids = list()
        actual_features = list()
        actual_labels = list()

        for ir in internal_references:
            material = css.query_material_by_internal_reference(ir)
            structure = material.structure()
            material_ids.append(material.id)
            feat, lab = self._actual_features(structure)
            actual_features.append(feat)
            actual_labels = lab

        # code under test
        fp = FeatureProvider(ENVIRONMENT)
        df = fp.get_featurization_block(material_ids,
                                        list(range(1, len(actual_features[0]))))

        # test labels
        columns = df.columns.values
        for i in range(len(columns)):
            self.assertEqual(columns[i], actual_labels[i])

        # test values
        for i in range(df.values.shape[0]):
            for j in range(df.values.shape[1]):
                self.assertEqual(df.values[i][j], actual_features[i][j])
