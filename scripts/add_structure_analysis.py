import os
import pickle
from camd import CAMD_S3_BUCKET
from camd.loop import Loop
from camd.analysis import AnalyzeStability_mod
from monty.os import cd, makedirs_p
from monty.serialization import loadfn, dumpfn
import boto3

from camd.analysis import AnalyzeStructures

def get_all_s3_folders():
    client = boto3.client('s3')
    result = client.list_objects(
        Bucket=CAMD_S3_BUCKET, Prefix="proto-dft/runs/", Delimiter='/')
    runs = [o.get('Prefix') for o in result.get('CommonPrefixes')]
    return runs


def sync_s3_folder(s3_folder, bucket=CAMD_S3_BUCKET,
                   local_folder=None):
    local_folder = local_folder or "."
    os.system('aws s3 sync s3://{}/{} {}'.format(
        bucket, s3_folder, local_folder))
    return True


def main():
    all_s3_prefixes = get_all_s3_folders()
    makedirs_p("cache")
    os.chdir("cache")
    print(all_s3_prefixes)
    for run in all_s3_prefixes:
        local_folder = run.split('/')[-2]
        sync_s3_folder(run, local_folder=local_folder)

if __name__ == "__main__":
    main()