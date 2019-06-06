"""
Module with class to manage feature access through database look up and through
computation.


"""
import logging
import pandas as pd
import numpy as np
from camd.database.access import CamdSchemaSession
from camd.model.feature.definition import feature_definition, index_sub_index, \
    feature_index_blocks, feature_directory, number_of_features, \
    directory_integrity_check
from camd.database.schema import PairwiseFeaturization, Feature


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
        self.camd_session = CamdSchemaSession(environment)
        self.feature_persister = FeaturePersister(self.camd_session)

        if not self.feature_persister.all_features_registered():
            logging.info('Registering new features in the database.')
            self.feature_persister.register_all_features()

    def get_featurization_block(self, material_ids, feature_ids,
                                output='pandas'):

        # check: features registered
        registered = self.camd_session.query_registered_feature_ids()
        for feature_id in feature_ids:
            if feature_id not in registered:
                raise ValueError(f'Requested feature_id {feature_id} is not ' +
                                 ' defined in codebase. ')

        # query: missing featurizations
        missing_pairwise = self.camd_session\
            .query_missing_featurizations(material_ids, feature_ids)

        # compute and persist: missing featurizations
        fc = FeatureComputer()
        for material_id, missing_feature_ids in missing_pairwise.items():

            # crude optimization decision here. If all features are missing,
            # compute all features together; otherwise individually
            material = self.camd_session.query_material(material_id)
            if len(missing_feature_ids) == len(feature_ids):
                all_feature_ids, feature_values = \
                    fc.compute_all_features(material)
            else:
                all_feature_ids = missing_feature_ids
                feature_values = list()
                for feature_id in missing_feature_ids:
                    feature_values += fc.compute(material, feature_id)

            # persist computed (missing) features
            self.feature_persister.persist_pairwise(material_id,
                                                    all_feature_ids,
                                                    feature_values)
            self.feature_persister.persist_feature_block(material_id)

        # query: full feature block
        sorted_material_ids, feature_arrays = self.camd_session\
            .query_full_feature_block(material_ids)

        feature_arrays = np.array(feature_arrays).astype(float)

        # down select: by feature_ids
        feature_arrays = feature_arrays[:, [x - 1 for x in feature_ids]]

        # format to desired output
        if output == 'numpy':
            return feature_arrays
        if output == 'pandas':
            column_names = self.camd_session.query_feature_labels(feature_ids)
            return pd.DataFrame(data=feature_arrays, columns=column_names)

        raise ValueError('Unsupported output format {output}')


class FeatureComputer:

    @staticmethod
    def compute(material, feature_id, return_all=False):
        """
        Computes a featurization for a material and feature_id.

        Args:
            material: camd.database.schema.Material
                CAMD Material object
            feature_id: int
                primary key value in camd.feature table
            return_all: bool, default=True
                flag select features that are computed alongside the one
                selected should be returned

        Returns: tuple
            list of featurization values
            optional
            list of all Featurizations computed alongside
            list of feature_labels
            list of feature_types

        """
        featurizer, feature_labels, types, sub_index = \
            feature_definition(feature_id)
        featurization_values = featurizer(material)

        if return_all:
            return featurization_values[sub_index], featurization_values, \
                   feature_labels, types
        else:
            return featurization_values[sub_index]

    @staticmethod
    def compute_all_features(material):
        """
        For a given material, returns feature values.

        Args:
            material: camd.database.schema.Material

        Returns:
            feature_ids: list
            feature_values: list

        """

        feature_values = list()
        feature_ids = list()
        for index in feature_index_blocks:
            f, l, t, sub_index = feature_definition(index)
            feature_values += [x[0] for x in f(material)]
            feature_ids += [index + x for x in range(len(l))]

        return feature_ids, feature_values


class FeaturePersister:

    def __init__(self, camd_session):
        self.camd_session = camd_session

    def all_features_registered(self):
        """
        Compares the number of features in the database and in the codebase.

        Returns: boolean
            True if number of features in database and codebase match
            False if number of features in database < codebase

        Raises:
            Exception if number of features in database > codebase

        """
        n_registered = self.camd_session.query_number_of_registered_features()
        n_coded = number_of_features()
        if n_registered > n_coded:
            logging.error(f'There are more features in the database ' +
                          '({n_registered}) than in the codebase ({n_coded}).')
            raise
        if n_registered < n_coded:
            return False
        return True

    def register_all_features(self):
        """
        Checks if all features are represented in the database feature table
        and if not, inserts them.

        Also, performs an integrity check on feature definition enumeration

        Returns: None

        """
        directory_integrity_check()
        for block_index in feature_index_blocks:
            block = feature_directory[block_index]
            for i in range(len(block['labels'])):
                id = block_index + i
                name = block['labels'][i]
                feature_type = block['types'][i]
                feature = Feature(id, name, feature_type)
                self.camd_session.insert(feature)

    def persist_pairwise(self, material_id, feature_ids, feature_values):
        """

        Args:
            material_id:
            feature_ids:
            feature_values:

        Returns:

        """
        self.camd_session.insert_pairwise_featurizations(material_id,
                                                         feature_ids,
                                                         feature_values)

    def persist_feature_block(self, material_id):
        feature_values = self.camd_session.query_all_featurizations(material_id)
        self.camd_session.upsert_block_featurization(material_id,
                                                     feature_values)
