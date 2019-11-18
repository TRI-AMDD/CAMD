import os
import pickle

from monty.os import makedirs_p, cd
from monty.serialization import loadfn
from camd.analysis import AnalyzeStability
from add_structure_analysis import get_all_s3_folders, sync_s3_folder


def process_run():
    folder = os.getcwd()
    if os.path.isfile("error.json"):
        error = loadfn("error.json")
        print("{} ERROR: {}".format(folder, error))

    required_files = ['seed_data.pickle']
    if not all([os.path.isfile(fn) for fn in required_files]):
        print("{} ERROR: no seed data, no analysis to be done")
    else:
        analyzer = AnalyzeStability(hull_distance=0.2, multiprocessing=True)
        with open(os.path.join("seed_data.pickle"), "rb") as f:
            result_df = pickle.load(f)
        # Hack to get new result ids
        all_result_ids = [mat_id for mat_id in result_df.index
                          if isinstance(mat_id, str)]
        output = analyzer.analyze(
            result_df, all_result_ids=all_result_ids,
            new_result_ids=all_result_ids)
        import nose; nose.tools.set_trace()


S3_SYNC = True


def main():
    if S3_SYNC:
        all_s3_prefixes = get_all_s3_folders()
        makedirs_p("cache")
        os.chdir("cache")
        print(all_s3_prefixes)
        for run in all_s3_prefixes:
            local_folder = run.split('/')[-2]
            sync_s3_folder(run, local_folder=local_folder)
    local_folders = os.listdir('cache')
    for local_folder in local_folders:
        with cd(os.path.join('cache', local_folder)):
            process_run()


if __name__ == "__main__":
    main()
