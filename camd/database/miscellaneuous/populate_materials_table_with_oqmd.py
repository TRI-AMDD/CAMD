"""

This is a helper script which can be run independently.

The script imports OQMD POSCAR files from S3 and creates CAMD database entries
in the materials table.

Location of OQMD files are hardcoded in this script.

"""

import logging, sys
import argparse
from sqlalchemy import exc

from camd.utils.s3 import iterate_bucket_items
from camd.database.access import CamdSchemaSession
from camd.database.schema import Material


OQMD_BUCKET_NAME = 'oqmd-chargedensity'
OQMD_PREFIX = 'OQMD_CHGCARs/'
POSCAR_SUFFIX = '_POSCAR'


def is_oqmd_poscar_key(key):
    """
    Checks if a key is following internal convention for being a POSCAR file.

    Args:
        key: str
            S3 key

    Returns: bool
        True if key is referencing a POSCAR file. False otherwise.
    """
    if key[0:13] != OQMD_PREFIX:
        return False
    if key[-7:] != POSCAR_SUFFIX:
        return False
    return True


def get_previously_imported_references(environment):
    """
    Returns a list of internal_references for materials that exist in the
    materials table,

    Args:
        environment: str
            Database environment (local, stage, production)

    Returns: list
        List of internal_references in the material table.
    """
    css = CamdSchemaSession(environment)
    return css.query_list_of_material_internal_references()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info('STARTING helper script to ingest OQMD POSCARS.')

    # db credentials and schema name in args
    parser = argparse.ArgumentParser(description='Provide database environment\
     as argument: (local, stage, production)')
    parser.add_argument('--env', '-e', type=str, help='-e environment')
    args = parser.parse_args()

    # hashset of already ingested internal references
    ingested = set(get_previously_imported_references(args.env))

    # ingestion
    logging.info(f'Connecting to database. Environment: {args.env}')
    css = CamdSchemaSession(args.env)

    logging.info(f'Starting ingestion.')
    i = 0
    batch = list()
    for item in iterate_bucket_items(bucket=OQMD_BUCKET_NAME):

        if is_oqmd_poscar_key(item['Key']):
            prefix = item['Key']
            if prefix not in ingested:
                material = Material.from_poscar_s3(OQMD_BUCKET_NAME, prefix,
                                                   internal_reference=prefix,
                                                   dft_computed=True)
                batch.append(material)
                i += 1
                if i % 1000 == 0:
                    success, exception = css.insert_batch(batch)
                    if success:
                        logging.info(f'Materials ingested: {i}')
                        batch = list()
                    else:
                        logging.error(f'{exception}')
                        sys.exit(1)

    success, exception = css.insert_batch(batch)
    if success:
        logging.info(f'Materials ingested: {i}')
    else:
        logging.error(f'{exception}')
        sys.exit(1)
    logging.info('COMPLETED helper script to ingest OQMD POSCARS.')
