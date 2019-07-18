# Computational Autonomy for Materials Discovery
# Copyright Toyota Research Institute 2019
import os


CAMD_ROOT = os.path.dirname(os.path.abspath(__file__))
CAMD_TEST_FILES = os.path.join(CAMD_ROOT, "tests", "test_files")
S3_CACHE = os.path.join(CAMD_ROOT, "s3_cache")

# Environment-based settings
TQDM_OFF = os.environ.get("TQDM_OFF", None)
CAMD_S3_BUCKET = os.environ.get("CAMD_S3_BUCKET", "camd-test")
CAMD_RUN_LOC = os.environ.get("CAMD_RUNS_DIR")

if TQDM_OFF:
    tqdm = lambda x: x
else:
    from tqdm import tqdm
