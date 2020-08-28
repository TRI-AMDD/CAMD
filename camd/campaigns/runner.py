#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Module and script for running the CAMD campaign runner

Usage:
    camd_runner CAMPAIGN_TYPE CAMPAIGN_NAME [options]

Options:
    --scratch            run in scratch directry
    -h --help            Show this screen
    --version            Show version

CAMPAIGN_TYPE should be a "type" for a campaign,
i.e. meta-agent or proto-dft-2, CAMPAIGN_ARG,
should be an input to the campaign e.g. Si-Ag
for chemsys or a reserved name for meta-agents
"""

import os
import tempfile
import shutil
from docopt import docopt
from camd.campaigns.meta_agent import MetaAgentCampaign
from camd.campaigns.structure_discovery import ProtoDFTCampaign


def main():
    """Utility function to run a reserved name campaign"""
    args = docopt(__doc__)
    campaign_type = args['CAMPAIGN_TYPE']
    campaign_name = args['CAMPAIGN_NAME']
    cwd = os.getcwd()
    if args['--scratch']:
        dirpath = tempfile.mkdtemp()
        os.chdir(dirpath)

    # Switch for different campaign types
    if campaign_type.startswith("proto-dft"):
        prefix = "{}/runs".format(campaign_type)
        campaign = ProtoDFTCampaign.from_chemsys(campaign_name, prefix=prefix)
    elif campaign_type == "meta_agent":
        campaign = MetaAgentCampaign.from_reserved_name(campaign_name)
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
