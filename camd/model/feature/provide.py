"""
Module with class to manage feature access through database look up and through
computation.


"""
import logging
from camd.database.access import CamdSchemaSession
from camd.model.feature.definition import feature_definition, index_sub_index, \
    feature_index_blocks, feature_directory, number_of_features
from camd.database.schema import Featurization, Feature


class FeatureProvider:
    """
    Class which handles provisioning of features for given materials.

    Featurization are looked up in the flow order:
        1. Look for feature/material combination in the database.
            If found, return
        2. Else, compute the feature(s) for the material(s)
        3. Store result in database
        4. Return feature value(s)

    """

    def __init__(self, environment):
        """
        Constructor. Choose run environment.

        Args:
            environment: str
                Database environment (local, stage, production)
        """
        self.camd_schema_session = CamdSchemaSession(environment)
        if not self._all_features_registered():
            logging.info('Registering new features in the database.')
            self._register_features()

    def _all_features_registered(self):
        """
        Compares the number of features in the database and in the codebase.

        Returns: boolean
            True if number of features in database and codebase match
            False if number of features in database < codebase

        Raises:
            Exception if number of features in database > codebase

        """
        n_registered = self.camd_schema_session\
            .query_number_of_registered_features()
        n_coded = number_of_features()
        if n_registered > n_coded:
            logging.error(f'There are more features in the database ' +
                          '({n_registered}) than in the codebase ({n_coded}).')
            raise
        if n_registered < n_coded:
            return False
        return True

    def _register_features(self):
        """
        Checks if all features are represented in the database feature table
        and if not, inserts them.

        Returns: None

        """
        for block_index in feature_index_blocks:
            block = feature_directory[block_index]
            for i in range(len(block['labels'])):
                id = block_index + i
                name = block['labels'][i]
                feature_type = block['types'][i]
                feature = Feature(id, name, feature_type)
                self.camd_schema_session.insert(feature)

    def one_featurization(self, material_id, feature_id):
        """
        Provides one feature for one material.

        Looks up the combination in the database and if not found computes it.

        Args:
            material_id: int
                primary key value in camd.material table
            feature_id: int
                primary key value in camd.feature table

        Returns:    numpy.array
            feature value as numpy array. Single value in array if numerical
            feature. One hot encoded, multi-value array in case of categorical
            feature type.

        """

        featurization = self.camd_schema_session\
            .query_featurization(material_id, feature_id)

        if featurization is None or featurization.value is None:
            fc = FeatureComputer()
            material = self.camd_schema_session.query_material(material_id)
            feature_value, all, all_labels, all_types = \
                fc.compute(material, feature_id)
            index, sub_index = index_sub_index(feature_id)
            for i in range(len(all)):
                new_feature_id = index + i
                new_label = all_labels[i]
                new_type = all_types[i]
                feature = self.camd_schema_session.query_feature(new_feature_id)
                if feature is None:
                    feature = Feature(new_feature_id, new_label, new_type)
                new_featurization = Featurization()
                new_featurization.material = material
                new_featurization.feature = feature
                new_featurization.value = all[i]

                self.camd_schema_session.session.add(new_featurization)
            self.camd_schema_session.session.commit()
            featurization = self.camd_schema_session\
                .query_featurization(material_id, feature_id)

        return featurization.value_array()

    def block_featurization(self, material_ids, feature_ids, output='pandas'):

        # identify missing featurizations
        # TODO

        # compute missing featurizations


        # query featurizations


        # format to output

        pass


class FeatureComputer:

    @staticmethod
    def compute(material, feature_id, return_all=True):
        """
        Computes a feature for a material.

        Args:
            material: camd.database.schema.Material
                CAMD Material object
            feature_id: int
                primary key value in camd.feature table
            return_all: bool, default=True
                flag select features that are computed alongside the one
                selected should be returned

        Returns: tuple
            camd.database.schema.Featurization
            list of all Featurizations computed alongside
            list of feature_labels
            list of feature_types

        """
        featurizer, feature_labels, types, sub_index = \
            feature_definition(feature_id)
        featurizations = featurizer(material)

        if return_all:
            return featurizations[sub_index], featurizations, feature_labels, \
                   types
        else:
            return featurizations[sub_index]

    @staticmethod
    def compute_all_features(material):

        featurizations = list()
        feature_labels = list()
        types = list()

        for block_index in feature_index_blocks:
            f, l, t, sub_index = feature_definition(block_index)
            l = l if isinstance(l, list) else [l]
            featurizations += [x[0] for x in f(material)]
            feature_labels += l
            types += t

        return featurizations, feature_labels, types

FeatureProvider('local')