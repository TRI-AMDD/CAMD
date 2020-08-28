#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.

import unittest
import subprocess
import boto3
import json
import time
import shlex
from multiprocessing import Pool
from camd import CAMD_S3_BUCKET
from camd.campaigns.worker import Worker


def teardown_s3():
    """Tear down test files in s3"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CAMD_S3_BUCKET)
    bucket.objects.filter(Prefix="{}".format("oqmd-atf")).delete()


@unittest.skipUnless(CAMD_S3_BUCKET, "CAMD S3 Bucket not set")
class WorkerTest(unittest.TestCase):
    def tearDown(self):
        teardown_s3()
        Worker("oqmd-atf").remove_stop_file()

    def submit_chemsyses(self, chemsyses):
        # Upload three things to s3
        s3_resource = boto3.resource("s3")
        for chemsys in chemsyses:
            key = "oqmd-atf/submit/{}/status.json".format(chemsys)
            obj = s3_resource.Object(CAMD_S3_BUCKET, key)
            obj.put(Body=json.dumps({"last_submitted": 10}))
            time.sleep(2)

    def put_runs(self, chemsyses):
        # Upload three things to s3
        s3_resource = boto3.resource("s3")
        for chemsys in chemsyses:
            key = "oqmd-atf/runs/{}/job_status.json".format(chemsys)
            obj = s3_resource.Object(CAMD_S3_BUCKET, key)
            obj.put(Body=json.dumps({"status": "started"}))

            key = "oqmd-atf/runs/{}/job_status.json".format(chemsys)
            obj = s3_resource.Object(CAMD_S3_BUCKET, key)
            obj.put(Body=json.dumps({"status": "started"}))

    def test_get_latest_chemsys(self):
        self.submit_chemsyses(["O-V", "O-Ti", "Fe-O"])
        worker = Worker("oqmd-atf")
        latest_chemsys = worker.get_latest_submission()
        self.assertEqual(latest_chemsys, "Fe-O")

        self.put_runs(["Fe-O"])
        latest_chemsys = worker.get_latest_submission()
        self.assertEqual(latest_chemsys, "O-Ti")

        self.put_runs(["O-V", "O-Ti"])
        latest_chemsys = worker.get_latest_submission()
        self.assertIsNone(latest_chemsys)

    def test_run_atf_campaign(self):
        self.submit_chemsyses(["O-Ti", "Fe-O"])
        worker = Worker("oqmd-atf")

        latest_chemsys = worker.get_latest_submission()
        self.assertEqual(latest_chemsys, "Fe-O")

        worker.start(num_loops=1)
        latest_chemsys = worker.get_latest_submission()
        self.assertEqual(latest_chemsys, "O-Ti")

        worker.start(num_loops=1)
        latest_chemsys = worker.get_latest_submission()
        self.assertIsNone(latest_chemsys)

    # TODO: This test is super slow, could make a
    #  new dummy campaign that executes with smaller overhead
    def test_stop(self):
        # Stopping a-priori
        worker = Worker("oqmd-atf")
        worker.write_stop_file()
        executed = worker.start()
        self.assertEqual(executed, 0)

        # # Ensure restarts after stop removal
        self.submit_chemsyses(["O-Ti", "Fe-O"])
        worker = Worker("oqmd-atf")
        worker.remove_stop_file()
        executed = worker.start(num_loops=2)
        self.assertEqual(executed, 2)

        # Ensure restarts after stop removal
        worker = Worker("oqmd-atf")
        worker.remove_stop_file()

        # TODO: this is a pretty hackish way of executing
        #  these in parallel and isn't guaranteed to work,
        #  but works for now

        with Pool(2) as p:
            result = p.map(worker_process, [0, 1])
        self.assertEquals(result[0], 1)

    def test_cli(self):
        self.submit_chemsyses(["O-Ti"])
        time.sleep(10)
        output = subprocess.check_output(
            "camd_worker start --campaign oqmd-atf --loops 1",
            shell=True
        )
        self.assertIn("Running experiments", output.decode('utf-8'))


def worker_process(index):
    if index == 0:
        print("index 0")
        worker = Worker("oqmd-atf")
        latest = worker.get_latest_submission()
        result = worker.start(sleep_time=7)
        print("returning {} {}".format(result, latest))
        return result
    else:
        time.sleep(3)
        worker = Worker("oqmd-atf")
        print("writing stop file")
        worker.write_stop_file()
        return None


if __name__ == '__main__':
    unittest.main()
