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
            logging.debug(f'IntegrityError: Object {camd_entity} not inserted.')
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

    def insert_pairwise_featurizations(self, material_id, feature_ids,
                                       feature_values):
        """
        For a given material_id and a set of feature_ids and respective values
        inserts pairwise featurization into the database.

        On unique constraint violations (does not insert; issues debug message)

        Args:
            material_id: int
            feature_ids: list
            feature_values: list

        Returns: bool, Exception
            True, None if batch insert successful
            False, exception if batch insert raised and exception

        """
        featurizations = [PairwiseFeaturization(material_id, feature_ids[i],
                                                feature_values[i])
                          for i in range(len(feature_ids))]
        for featurization in featurizations:
            self.insert(featurization)

    def upsert_block_featurization(self, material_id, feature_values):
        """
        Inserts feature value array into block_featurization table.
        Overrides, if respective material_id already has values.

        Args:
            material_id: int
            feature_values: list

        Returns: None

        """
        conn = self.session.connection()
        n_features = len(feature_values)
        sql = """INSERT INTO camd.block_featurization 
        (material_id, n_features, feature_values) 
        VALUES (:material_id, :n_features, :feature_values) 
        ON CONFLICT (material_id) DO UPDATE SET 
        n_features = EXCLUDED.n_features, 
        feature_values = EXCLUDED.feature_values"""
        s = text(sql)
        conn.execute(s, material_id=material_id, n_features=n_features,
                     feature_values=feature_values)
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
            CAMD Featurization object, if exists. None otherwise.

        """
        return self.session.query(PairwiseFeaturization)\
            .filter_by(feature_id=feature_id, material_id=material_id).first()

    def query_featurizations(self, material_ids, feature_ids):
        """
        Queries the database for featurizations of the given material and
        feature ids and returns them.

        Args:
            material_ids: list
            feature_ids: list

        Returns: dict
            Map of material_ids to list of feature values

        """
        conn = self.session.connection()
        sql = """SELECT material_id, feature_id, value FROM camd.featurization 
        WHERE material_id IN :material_id_list 
        AND feature_id IN :feature_id_list ORDER BY 1, 2"""
        s = text(sql)
        result = conn.execute(s, feature_id_list=tuple(feature_ids),
                              material_id_list=tuple(material_ids))
        feature_values = dict()
        for row in result:
            vals = feature_values.get(row[0], [])
            vals += [float(x) for x in list(row[2])]
            feature_values[row[0]] = vals
        return feature_values

    def query_all_featurizations(self, material_id):
        """
        Returns a list of all feature values for a material ordered by
        feature_id.

        Args:
            material_id: int

        Returns: list
            featurization values for material

        """
        sql = """SELECT feature_id, value FROM camd.pairwise_featurization 
        WHERE material_id=:material_id ORDER BY 1"""
        s = text(sql)
        conn = self.session.connection()
        result = conn.execute(s, material_id=material_id)
        values = list()
        for row in result:
            values.append(row[1])
        return values

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

    def query_registered_feature_ids(self):
        """
        Returns all feature_ids which exists in the Feature table.

        Returns: list
            List of feature ids.

        """
        result = self.session.query(Feature.id).all()
        feature_ids = [row[0] for row in result]
        return feature_ids

    def query_missing_featurizations(self, material_ids, feature_ids):
        """
        Queries database for missing combinations of material and feature ids
        in featurization table.

        Args:
            material_ids: list
            feature_ids: list

        Returns: dict
            map of material_ids to lists of missing feature_ids

        """
        conn = self.session.connection()
        sql = """SELECT t1.material_id, t1.feature_id 
        FROM (SELECT f.id AS feature_id, m.id AS material_id 
        FROM (SELECT id FROM camd.feature WHERE id in :feature_id_list) AS f, 
        (SELECT id FROM camd.material WHERE id in :material_id_list) AS m) t1 
        left JOIN (SELECT cfz.material_id, cf.id AS feature_id 
        FROM camd.feature cf 
        left join (SELECT material_id, feature_id 
        FROM camd.pairwise_featurization 
        WHERE material_id in :material_id_list) cfz 
        on cf.id=cfz.feature_id) t2 
        on t1.material_id=t2.material_id AND t1.feature_id=t2.feature_id 
        WHERE t1.material_id IS NULL or t2.feature_id IS NULL ORDER BY 1, 2"""
        s = text(sql)
        result = conn.execute(s, feature_id_list=tuple(feature_ids),
                              material_id_list=tuple(material_ids))
        missing_featurization = dict()
        for row in result:
            f_ids = missing_featurization.get(row[0], [])
            f_ids.append(row[1])
            missing_featurization[row[0]] = f_ids
        return missing_featurization

    def query_feature_labels(self, feature_ids):
        """
        Returns the feature labels for the provided feature_ids

        Args:
            feature_ids: list

        Returns: list
            List of feature labels.

        """
        query = self.session.query(Feature.name)\
            .filter(Feature.id.in_(feature_ids))
        results = query.all()
        feature_labels = list()
        for row in results:
            feature_labels.append(row[0])
        return feature_labels

    def query_full_feature_block(self, material_ids):
        """
        For given material_ids, returns matrix of featurization values.

        Args:
            material_ids: list

        Returns: list
            list of list with featurization values

        """
        bfs = self.session\
            .query(BlockFeaturization)\
            .filter(BlockFeaturization.material_id.in_(material_ids))\
            .order_by(BlockFeaturization.material_id.asc())\
            .all()
        sorted_material_ids = [bf.material_id for bf in bfs]
        feature_arrays = [bf.feature_values for bf in bfs]
        return sorted_material_ids, feature_arrays

    def oqmd_ids_to_material_ids(self, oqmd_ids):
        """
        Looks up the respective internal material_ids for a list of oqmd_ids.

        Args:
            oqmd_ids: list

        Returns: list
            Corresponding material_ids

        """
        internal_references = [f'OQMD_CHGCARs/{x}_POSCAR' for x in oqmd_ids]
        sql = """SELECT id, internal_reference FROM camd.material 
        WHERE internal_reference IN :internal_references"""
        s = text(sql)
        conn = self.session.connection()
        result = conn.execute(s, internal_references=tuple(internal_references))
        material_id_map = dict()
        for row in result:
            material_id_map[row[1]] = row[0]
        material_ids = list()
        for ir in internal_references:
            material_ids.append(material_id_map[ir])
        return material_ids

