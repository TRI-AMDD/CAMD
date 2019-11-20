#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.agent.table import AGENT_PARAMS
import pickle
import boto3


def initialize_agent_campaign(name, dataframe, agent_pool=None):
    """
    Quick function to initialize agent stability campaign

    Args:
        name (name): name of the campaign to initialize
        dataframe (DataFrame): dataframe to use for sampling agents
        agent_pool (ParameterTable): parameter table of agents

    Returns:
        None

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    result = client.list_objects(Bucket=CAMD_S3_BUCKET, prefix=prefix)
    if result:
        raise ValueError(
            "{} exists in s3, please choose another name".format(name))

    agent_pool = agent_pool or ParameterTable(AGENT_PARAMS)
    pickled_agent = pickle.dumps(agent_pool)
    client.Object(CAMD_S3_BUCKET, "{}/agent_pool.pickle").put(
        Body=pickled_agent)

    pickled_dataframe = pickle.dumps(dataframe)
    client.Object(CAMD_S3_BUCKET, "{}/atf_data.pickle").put(
        Body=pickled_dataframe)


def update_agent_pool(name, params):
    """
    Function to update the agent pool associated with a campaign

    Args:
        name (str): name of campaign
        params ([{}]): list of dicts associated with agent
            configuration parameters

    Returns:
        None

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    result = client.list_objects(Bucket=CAMD_S3_BUCKET, prefix=prefix)
    if not result:
        raise ValueError(
            "{} does not exist in s3, cannot update agent pool".format(name))
    raw_agent_pool = client.Object(
        CAMD_S3_BUCKET, "{}/agent_pool.pickle").get()['Body']
    agent_pool = pickle.loads(raw_agent_pool)
    agent_pool.append(params)
    pickled_agent = pickle.dumps(agent_pool)
    client.Object(CAMD_S3_BUCKET, "{}/agent_pool.pickle").put(
        Body=pickled_agent)
