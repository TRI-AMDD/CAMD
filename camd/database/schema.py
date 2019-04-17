"""
Representing the data model as python objects
(SQLAlchemy)

Individual classes / entities have additional methods to provide specific
functionality.

"""

import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import relation
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql
from abc import ABCMeta
from pymatgen import Structure
from camd.utils.s3 import read_s3_object_body


Base = declarative_base()


class CamdEntity:
    """
    Interface for objects represented in the CAMD schema.

    """
    __metaclass__ = ABCMeta


class Material(Base, CamdEntity):
    __tablename__ = 'material'

    id = Column(Integer, primary_key=True, autoincrement=True)
    internal_reference = Column(String, unique=True)
    definition = Column(JSON)
    poscar = Column(Text)
    dft_computed = Column(Boolean, nullable=False)

    def __init__(self, dft_computed, internal_reference, definition, poscar):
        self.internal_reference = internal_reference
        self.definition = definition
        self.poscar = poscar
        self.dft_computed = dft_computed

    def __repr__(self):
        return "Material(%r, %r, %r)" % (self.id, self.internal_reference,
                                         self.dft_computed)

    def structure(self):
        if self.poscar is not None:
            return Structure.from_str(self.poscar, fmt='poscar')
        return None

    @staticmethod
    def from_poscar(text, internal_reference=None, dft_computed=True):
        """
        Static method to create a Material from poscar text format.

        Args:
            text:   str
                poscar text
            internal_reference: str
                material name
            dft_computed: bool
                flag for dft computation done on this material

        Returns: camd.database.schema.Material
            CAMD Material object

        """
        return Material(internal_reference=internal_reference,
                        definition=None, poscar=text, dft_computed=dft_computed)

    @staticmethod
    def from_poscar_file(file_path, internal_reference=None, dft_computed=True):
        """
        Static method to create a Material from local file.

        Args:
            file_path: str
                local file path
            internal_reference: str
                material name
            dft_computed: bool
                flag for dft computation done on this material

        Returns: camd.database.schema.Material
            CAMD Material object

        """

        with open(file_path) as f:
            poscar = f.read()
        definition = None

        return Material(internal_reference=internal_reference,
                        definition=definition, poscar=poscar,
                        dft_computed=dft_computed)

    @staticmethod
    def from_poscar_s3(bucket, prefix, internal_reference=None,
                       dft_computed=True):
        """
        Static method to create a Material from S3 object

        Args:
            bucket: str
                S3 bucket
            prefix: str
                S3 prefix
            internal_reference: str
                material name
            dft_computed: bool
                flag for dft computation done on this material

        Returns: camd.database.schema.Material
            CAMD Material object

        """
        content = read_s3_object_body(bucket, prefix).decode('utf8')
        return Material.from_poscar(content, internal_reference,
                                    dft_computed=dft_computed)


class Feature(Base, CamdEntity):
    __tablename__ = 'feature'

    id = Column(Integer, primary_key=True, autoincrement=False)
    name = Column(String, unique=True, nullable=False)
    feature_type = Column('feature_type', String, nullable=False)
    possible_values = Column(postgresql.ARRAY(String, dimensions=1))

    def __init__(self, id, name, feature_type, possible_values=None):
        if feature_type not in ('numerical', 'categorical'):
            raise ValueError("""feature_type can only be either numerical or 
            categorical. Got %s""" % feature_type)
        self.id = id
        self.name = name
        self.feature_type = feature_type
        self.possible_values = possible_values

    def __repr__(self):
        return "Feature(%r, %r, %r, %r)" % (self.id, self.name,
                                            self.feature_type,
                                            self.possible_values)


class Featurization(Base, CamdEntity):
    __tablename__ = 'featurization'

    material_id = Column(Integer, ForeignKey('material.id'), primary_key=True,
                         autoincrement=False)
    feature_id = Column(Integer, ForeignKey('feature.id'), primary_key=True,
                        autoincrement=False)
    value = Column(postgresql.ARRAY(DECIMAL, dimensions=1))

    material = relation("Material", backref='material', lazy=False)
    feature = relation("Feature", backref='feature', lazy=False)

    def __repr__(self):
        return "Featurization(%r, %r, %r)" % (self.material_id, self.feature_id,
                                        self.value)

    def value_array(self):
        """
        Provides access to the featurization in numpy array format.

        Returns: numpy.array
            1d array of features

        """
        return np.array([float(x) for x in self.value])