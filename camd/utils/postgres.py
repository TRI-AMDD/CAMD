"""
Module to provide connection objects to various postgres environments.

Current environments:
- local     Local development environment. Non-secret credentials.


"""

import psycopg2
from psycopg2.sql import SQL, Identifier, Placeholder, Composed
import sqlalchemy


SCHEMA_CAMD = 'camd'


def postgres_credentials(environment):
    """
    Looks up and returns the respective credentials for the database
    environment.


    Args:
        environment:    str representing the database environment
                        (e.g. local, stage, production)

    Returns:
        credentials     dict with credentials

    """

    credentials = dict()

    if environment == 'local':
        credentials['host'] = 'localhost'
        credentials['username'] = 'localuser'
        credentials['port'] = 5432
        credentials['password'] = 'localpassword'
        credentials['dbname'] = 'local'

    if environment == 'stage':
        raise NotImplementedError()

    return credentials


def postgres_connection(environment):
    """
    Establishes a database connection to the selected environment.
    Database credentials are obtained through the respective means.

    Args:
        environment:    str representing the database environment
                        (e.g. local, stage, production)

    Returns:
        connection, cursor pair for the database connection

    """

    credentials = postgres_credentials(environment)
    connection = psycopg2.connect(host=credentials['host'],
                                  user=credentials['username'],
                                  port=credentials['port'],
                                  password=credentials['password'],
                                  dbname=credentials['dbname'])
    cursor = connection.cursor()
    return connection, cursor


def database_available(environment):
    """
    Checks if a database is available in the selected environment.

    Args:
        environment:    str representing the database environment
                        (e.g. local, stage, production)

    Returns:
        boolean         True if database is available. False otherwise.

    """

    credentials = postgres_credentials(environment)

    try:
        psycopg2.connect(host=credentials['host'],
                         user=credentials['username'],
                         port=credentials['port'],
                         password=credentials['password'],
                         dbname=credentials['dbname'],
                         connect_timeout=3)
    except psycopg2.OperationalError:
        return False
    else:
        return True


def sqlalchemy_engine(environment, schema='default'):
    """
    Provides a sqlalchemy engine object for the selected environment.

    Args:
        environment: str
            name of the environment (local, stage)
        schema: str
            name of database schema (default='default' --> camd)

    Returns: sqlalchemy.engine.Engine
        A sqlalchemy engine for the environment
    """
    schema_name = SCHEMA_CAMD if schema == 'default' else schema
    credentials = postgres_credentials(environment)
    host = credentials['host']
    username = credentials['username']
    port = credentials['port']
    pwd = credentials['password']
    dbname = credentials['dbname']
    connection_string = f'postgres://{username}:{pwd}@{host}:{port}/{dbname}'
    connect_args = {'options': '-csearch_path={}'.format(schema_name)}
    engine = sqlalchemy.create_engine(connection_string,
                                      connect_args=connect_args)
    return engine


def sqlalchemy_session(environment, schema='default'):
    """
    Creates and returns a session object for the selected environment.

    Args:
        environment: str
            name of the environment (local, stage)
        schema: str
            name of database schema (default='default' --> camd)

    Returns: sqlalchemy.orm.session.session
        A sqlalchemy session for the environment

    """
    engine = sqlalchemy_engine(environment)
    schema_name = SCHEMA_CAMD if schema == 'default' else schema
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    session.execute(f'SET search_path TO {schema_name};')
    return session


class PostgresConnection:
    """

    Manages a database connection and has custome insert and query operations
    applicable to specific use cases.

    The database connection is opened upon creation of a PostgresConnection
    object and closes upon being garbage collected.

    """

    def __init__(self, environment):
        """
        Class constructor. Opens database connection.

        Args:
            environment: str
                the database environment (e.g. local, stage, production)
        """
        self.connection, self.cursor = postgres_connection(environment)

    def custom_insert(self, schema, table, field_value_pairs):
        """
        Executes and commits a single record sql insert of field/value pairs

        Args:
            schema: str
                name of postgres database schema
            table: str
                name of database table
            field_value_pairs: dict
                Key/Value pairs of field values to be inserted.

        Returns:
            None

        """
        field_names = list()
        field_values = list()
        for k, v in field_value_pairs.items():
            field_names.append(k)
            field_values.append(v)
        sql_insert = SQL('INSERT INTO {}.{} ({}) VALUES ({});')\
            .format(Identifier(schema),
                    Identifier(table),
                    SQL(', ').join(map(Identifier, field_names)),
                    SQL(', ').join(Placeholder() * len(field_values)))\
            .as_string(self.connection)
        self.cursor.execute(sql_insert, field_values)
        self.connection.commit()

    def custom_select(self, schema, table, select_fields, conditions=None,
                      order_by=[]):
        """
        Runs a select query for selected fields and applies conditional
        statement.

        Args:
            schema: str
                name of postgres database schema
            table: str
                name of database table
            select_fields: list
                names of fields to be included in the selection
            conditions: dict
                contains key/value pairs of field name and value. Will be used
                to construct a WHERE clause in which all fields have to be equal
                their respective value in this dict.

        Returns: dict
            query result structured as dict which contains a list of data for
            each field name key

        """

        # query construction and execution
        sql_start = SQL('SELECT {} FROM {}.{}')\
            .format(SQL(', ').join(map(Identifier, select_fields)),
                    Identifier(schema),
                    Identifier(table))
        if conditions is None or len(conditions) == 0:
            sql = Composed([sql_start, SQL(';')]).as_string(self.connection)
            self.cursor.execute(sql)
        else:
            condition_fields = list()
            condition_values = list()
            for k, v in conditions.items():
                condition_fields.append(k)
                condition_values.append(v)
            sql_components = [sql_start, SQL(' WHERE')]
            for i in range(len(condition_fields)):
                if condition_values[i] is None:
                    sql_components.append(SQL(' {} IS NULL') \
                        .format(
                        Identifier(condition_fields[i])))
                else:
                    sql_components.append(SQL(' {}={}')\
                                          .format(
                        Identifier(condition_fields[i]),
                        Placeholder()))
                if i < len(condition_fields) - 1:
                    sql_components.append(SQL(' AND'))
                else:
                    sql_components.append(SQL(';'))
            sql = Composed(sql_components).as_string(self.connection)
            while None in condition_values:
                condition_values.remove(None)
            self.cursor.execute(sql, condition_values)

        # fetch results
        result = self.cursor.fetchall()

        # format result to dictionary
        data = dict()
        for field in select_fields:
            data[field] = list()
        for r in result:
            for i in range(len(select_fields)):
                data[select_fields[i]].append(r[i])

        return data

    def pandas_insert(self, df, schema, table, on_conflict='do_nothing'):
        """
        Inserts a data frame into a table in the schema. Assumes that the
        columns in the data frame match the columns in the database table.
        Does not check.

        Args:
            df: pandas.DataFrame
                data frame object to be inserted
            schema: str
                postgres schema name
            table: str
                postgres table name
            on_conflict: str
                on conflict command if constraint is violated
                (default: do_nothing)

        Returns:
            void

        """
        if df is None:
            return

        if on_conflict == 'do_nothing':
            data = df.values
            placeholder_str = '(' + ','.join(['%s'] * len(data[0])) + ')'
            args_str = ','.join(self.cursor.mogrify(placeholder_str, x)\
                                .decode('ascii') for x in data)
            sql_insert = \
                SQL('INSERT INTO {}.{} VALUES %s ON CONFLICT DO NOTHING;')\
                .format(Identifier(schema), Identifier(table))\
                .as_string(self.connection)
            self.cursor.execute(sql_insert % (args_str,))
            self.connection.commit()
        elif on_conflict == 'upsert':
            raise NotImplementedError()

    def close(self):
        self.connection.close()

    def __del__(self):
        self.close()
