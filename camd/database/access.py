"""
Module for class which connects with CAMD postgres schema to facilitate
inserts and queries custom to CAMD.

"""

from sqlalchemy import exc
from camd.database.schema import *
from camd.utils.postgres import sqlalchemy_session


class CamdSchemaSession:
    """
    Class which uses a sqlalchemy session to interact with the CAMD database
    schema.

    Session is open as long as the instantiated object is in memory and closed
    when garbage collected.

    """

    def __init__(self, environment):
        """
        Constructor. Choose run environment.

        Args:
            environment: str
                Database environment (local, stage, production)
        """
        self.session = sqlalchemy_session(environment)

    def insert(self, camd_entity):
        """
        Adds a CAMD entity into the database. A CAMD entity is an object which
        is represented in the CAMD database schema.

        Args:
            camd_entity: camd.database.schema.CamdEntity

        Returns:
            None

        """
        if not isinstance(camd_entity, CamdEntity):
            raise ValueError('Object must be of type CamdEntity.')

        try:
            self.session.add(camd_entity)
            self.session.commit()
        except exc.IntegrityError as ie:
            logging.warning(f'IntegrityError: Object not inserted.')
            self.session.rollback()

    def insert_batch(self, camd_entities):
        """
        Adds a list of CAMD entities to the database.

        Args:
            camd_entities: list
                list of camd.database.schema.CamdEntity

        Returns: bool, Exception
            True, None if batch insert successful
            False, exception if batch insert raised and exception
        """
        try:
            self.session.bulk_save_objects(camd_entities)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            return False, e

        return True, None

    def query_featurization(self, material_id, feature_id):
        """
        Queries a featurization entry by material_id and feature_id

        Args:
            material_id: int
                primary key value in camd.material table
            feature_id: int
                primary key value in camd.feature table

        Returns: camd.database.schema.Featurization
            CAMD Featurization object, if exists. None otherwise.

        """
        return self.session.query(Featurization)\
            .filter_by(feature_id=feature_id, material_id=material_id).first()

    def query_material(self, material_id):
        """
        Queries a material entry by material_id

        Args:
            material_id: int
                primary key value in camd.material table

        Returns: camd.database.schema.Material
            CAMD Material object, if exists. None otherwise.

        """
        return self.session.query(Material).filter_by(id=material_id).first()

    def query_feature(self, feature_id):
        """
        Queries a feature entry by feature_id

        Args:
            feature_id: int
                primary key value in camd.feature table

        Returns: camd.database.schema.Feature
            CAMD Feature object, if exists. None otherwise.

        """
        return self.session.query(Feature).filter_by(id=feature_id).first()

    def query_material_by_internal_reference(self, internal_reference):
        """
        Queries a material entry by internal reference.

        Args:
            internal_reference: str
                unique internal reference in camd.material table

        Returns: camd.database.schema.Material
            CAMD Material object, if exists. None otherwise.

        """
        return self.session.query(Material)\
            .filter_by(internal_reference=internal_reference).first()

    def query_list_of_material_internal_references(self):
        """
        Returns a list of internal_references for materials that exist in the
        materials table,

        Args:
            self

        Returns: list
            List of internal_references in the material table.
        """
        results = self.session.query(Material.internal_reference).all()
        return [value[0] for value in results]

    def query_number_of_registered_features(self):
        """
        Returns the number of records in the feature table.

        Returns: int
            number of registered features

        """
        result = self.session.query(Feature).count()
        return result
