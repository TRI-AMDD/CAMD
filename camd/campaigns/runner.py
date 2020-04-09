#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Module and script for running the CAMD campaign runner

Usage:
    camd_runner CAMPAIGN_PREFIX [options]

Options:
    --scratch            run in scratch directry
    -h --help            Show this screen
    --version            Show version

CAMPAIGN_PREFIX should be a "reserved name" for a campaign,
i.e. '/'-delimited string beginning with the campaign type
and reserved name, e. g. meta-agent/test-1
"""

import os
import tempfile
import shutil
from docopt import docopt
from camd.campaigns.meta_agent import MetaAgentCampaign


def main():
    """Utility function to run a reserved name campaign"""
    args = docopt(__doc__)
    campaign_prefix = args['CAMPAIGN_PREFIX']
    campaign_type, reserved_name = campaign_prefix.split('/', 1)
    cwd = os.getcwd()
    if args['--scratch']:
        dirpath = tempfile.mkdtemp()
        os.chdir(dirpath)

    # Switch for different campaign types
    if campaign_type == "meta_agent":
        campaign = MetaAgentCampaign.from_reserved_name(reserved_name)
    else:
        raise ValueError("{} is not a supported campaign type".format(
            campaign_type))

    campaign.autorun()

    # Cleanup
    if args['--scratch']:
        os.chdir(cwd)
        shutil.rmtree(dirpath)


if __name__ == "__main__":
    main()
