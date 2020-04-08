#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
Module and script for running the CAMD campaign runner

Usage:
    camd_runner CAMPAIGN_PREFIX

Options:
    -h --help            Show this screen
    --version            Show version

CAMPAIGN_PREFIX should be a "reserved name" for a campaign,
i.e. '/'-delimited string beginning with the campaign type
and reserved name, e. g. meta-agent/test-1
"""
from docopt import docopt
from camd.campaigns.meta_agent import MetaAgentCampaign


def main():
    """Utility function to run a reserved name campaign"""
    args = docopt(__doc__)
    campaign_prefix = args['CAMPAIGN_PREFIX']
    campaign_type, reserved_name = campaign_prefix.split('/', 1)
    # Switch for different campaign types
    if campaign_type == "meta-agent":
        campaign = MetaAgentCampaign.from_reserved_name(reserved_name)
    else:
        raise ValueError("{} is not a supported campaign type".format(
            campaign_type))

    campaign.autorun()


if __name__ == "__main__":
    main()
