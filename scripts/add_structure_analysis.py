import os
from camd import CAMD_S3_BUCKET
from monty.os import makedirs_p
import boto3
from camd.analysis import update_run_w_structure


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


def update_s3(s3_folder, bucket=CAMD_S3_BUCKET,
              local_folder=None):
    local_folder = local_folder or "."
    os.system('aws s3 sync {} s3://{}/{}'.format(
        local_folder, bucket, s3_folder))


def main():
    all_s3_prefixes = get_all_s3_folders()
    makedirs_p("cache")
    os.chdir("cache")
    print(list(enumerate(all_s3_prefixes)))
    for run in all_s3_prefixes[27:]:
        local_folder = run.split('/')[-2]
        sync_s3_folder(run, local_folder=local_folder)
        update_run_w_structure(local_folder)
        update_s3(run, local_folder=local_folder)


if __name__ == "__main__":
    main()
