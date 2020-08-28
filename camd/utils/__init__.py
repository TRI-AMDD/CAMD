"""
Module for generic utilities for data munging, s3 interaction etc.
"""
import datetime
import re


def get_new_version(current_ver):
    """
    Helper function to increment semantic versioning by date

    Args:
        current_ver (str): string-formatted date as YYYY.MM.DD,
            or YYYY.MM.DD-post{NUMBER} for post-versions released
            on the same day

    Returns:
        (str) new version string

    """
    today = datetime.datetime.today().strftime("%Y.%-m.%-d")

    # Extract current_ver_date
    current_ver_date = re.sub(r"-post\d+", "", current_ver)

    if today == current_ver_date:
        if "post" in current_ver:
            # Increment post by 1
            new_ver = re.sub(r"post(\d+)", lambda exp: "post{}".format(int(exp.group(1)) + 1), current_ver)
        else:
            # Append "-post0" to version
            new_ver = "{}-post0".format(current_ver)
    else:
        # Set new version as today
        new_ver = today

    return new_ver
