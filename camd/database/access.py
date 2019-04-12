"""
Module for class which connects with CAMD postgres schema to facilitate
inserts and queries custom to CAMD.

"""

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

        self.session.add(camd_entity)
        self.session.commit()

    def query_featurization(self, material_id, feature_id):
        """
        Queries a featurization entry by material_id and feature_id

        Args:
            material_id: int
                primary key value in camd.material table
            feature_id: int
                primary key value in camd.feature table

        Returns: camd.database.schema.Featurization
            CAMD Featurization object

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
            CAMD Material object

        """
        return self.session.query(Material).filter_by(id=material_id).first()
