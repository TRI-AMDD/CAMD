import os
import pickle
import requests
import json

import pandas as pd
from tqdm import tqdm
from monty.os import makedirs_p, cd
from monty.serialization import loadfn
from camd.analysis import AnalyzeStability
from add_structure_analysis import get_all_s3_folders, sync_s3_folder

S3_SYNC = False
API_URL = "http://camd-api.matr.io"
CATCH_ERRORS = True


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

        unique_structures = loadfn("discovered_unique_structures.json")
        all_result_ids = list(unique_structures.keys())

        summary = result_df.loc[all_result_ids]
        summary = summary[['Composition', 'delta_e']]
        analyzer.analyze(
            result_df, all_result_ids=all_result_ids,
            new_result_ids=all_result_ids)
        # Add stabilities
        summary['stabilities'] = pd.Series(analyzer.stabilities)

        chemsys = os.path.split(folder)[-1]
        # Get all DFT data
        response = requests.get('{}/synthesis-discovery/{}/dft-results'.format(
            API_URL, chemsys))
        data = json.loads(response.content.decode('utf-8'))
        data = pd.DataFrame(data)
        aggregated = {}
        for result in data['dft_results']:
            aggregated.update(result)
        simulation_data = pd.DataFrame.from_dict(aggregated, orient='index')
        summary['bandgap'] = simulation_data['bandgap']
        # Apply garcia correction
        summary['bandgap_garcia_exp'] = 1.358 * summary['bandgap'] + 0.904
        return summary


def main():
    if S3_SYNC:
        all_s3_prefixes = get_all_s3_folders()
        makedirs_p("cache")
        os.chdir("cache")
        print(all_s3_prefixes)
        for run in all_s3_prefixes:
            local_folder = run.split('/')[-2]
            sync_s3_folder(run, local_folder=local_folder)
    all_dfs = []
    problem_folders = []

    local_folders = os.listdir('cache')
    # local_folders = ['Mn-S']
    for local_folder in tqdm(local_folders):
        with cd(os.path.join('cache', local_folder)):
            if CATCH_ERRORS:
                try:
                    all_dfs.append(process_run())
                except Exception as e:
                    print(e)
                    problem_folders.append(local_folder)
            else:
                all_dfs.append(process_run())
    output = pd.concat(all_dfs, axis=0)
    output = output.sort_values('stabilities')
    output.to_csv("summary.csv")
    import nose; nose.tools.set_trace()
    print("problems:")
    print(problem_folders)


if __name__ == "__main__":
    main()
