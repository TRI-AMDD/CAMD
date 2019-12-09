#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.agent.meta import AGENT_PARAMS, RandomMetaAgent, \
    convert_parameter_table_to_dataframe
from camd.experiment.atf import LocalAgentSimulation
from camd.loop import Campaign
import pickle
import boto3
import botocore


def initialize_agent_campaign(name, dataframe, agent_pool=None,
                              bucket=CAMD_S3_BUCKET):
    """
    Quick function to initialize agent stability campaign

    Args:
        name (name): name of the campaign to initialize
        dataframe (DataFrame): dataframe to use for sampling agents
        agent_pool (ParameterTable): parameter table of agents
        bucket (str): name of bucket to use

    Returns:
        None

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    result = client.list_objects(Bucket=bucket, Prefix=prefix)
    if result.get('Contents'):
        raise ValueError(
            "{} exists in s3, please choose another name".format(name))

    agent_pool = agent_pool or ParameterTable(AGENT_PARAMS)
    pickled_agent = pickle.dumps(agent_pool)
    client.put_object(
        Bucket=bucket,
        Key="{}/agent_pool.pickle".format(prefix),
        Body=pickled_agent
    )

    pickled_dataframe = pickle.dumps(dataframe)
    client.put_object(
        Bucket=bucket,
        Key="{}/atf_data.pickle".format(prefix),
        Body=pickled_dataframe
    )


def update_agent_pool(name, params, bucket=CAMD_S3_BUCKET):
    """
    Function to update the agent pool associated with a campaign

    Args:
        name (str): name of campaign
        params ([{}]): list of dicts associated with agent
            configuration parameters
        bucket (str): name of bucket to update

    Returns:
        None

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    agent_pool = load_agent_pool(name, bucket)
    agent_pool.extend(params)

    pickled_agent = pickle.dumps(agent_pool)
    client.put_object(
        Bucket=bucket,
        Key="{}/agent_pool.pickle".format(prefix),
        Body=pickled_agent
    )


def load_agent_pool(name, bucket=CAMD_S3_BUCKET):
    """
    Loads an agent pool

    Args:
        name (str): name of campaign
        bucket (str): name of bucket

    Returns:
        (ParameterTable): parameter table of agents

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    try:
        raw_agent_pool = client.get_object(
            Bucket=bucket,
            Key="{}/agent_pool.pickle".format(prefix),
        )['Body']
    except botocore.exceptions.ClientError as e:
        raise ValueError(
            "{} does not exist in s3, cannot update agent pool".format(name)
        )
    agent_pool = pickle.loads(raw_agent_pool.read())
    return agent_pool


def load_atf_data(name, bucket=CAMD_S3_BUCKET):
    """
    Loads an agent pool

    Args:
        name (str): name of campaign
        bucket (str): name of bucket

    Returns:
        (ParameterTable): parameter table of agents

    """
    client = boto3.client("s3")
    prefix = "agent_testing/{}".format(name)
    try:
        raw_agent_pool = client.get_object(
            Bucket=bucket,
            Key="{}/atf_data.pickle".format(prefix),
        )['Body']
    except botocore.exceptions.ClientError as e:
        raise ValueError(
            "{} does not exist in s3, cannot load atf_data".format(name)
        )
    agent_pool = pickle.loads(raw_agent_pool.read())
    return agent_pool


def run_meta_agent_campaign(name, meta_agent=None, bucket=CAMD_S3_BUCKET,
                            n_iterations=5):
    """
    Run a meta-agent campaign to explore agent space
    for a given dataset

    Args:
        name (str): name of campaign (as defined above)
        meta_agent (HypothesisAgent): meta-agent for selecting
            the next agents to test
        bucket (str): s3 bucket name for syncing and pulling data
        n_iterations (int): number of iterations for which to
            run the loop

    Returns:
        None

    """
    agent_pool = load_agent_pool(name, bucket)
    atf_data = load_atf_data(name, bucket)
    meta_agent = meta_agent or RandomMetaAgent(
        agent_pool=agent_pool, n_query=1)
    experiment = LocalAgentSimulation(
        atf_dataframe=atf_data, analyzer=None,
        iterations=50, n_seed=1)
    candidate_data = convert_parameter_table_to_dataframe(agent_pool)
    loop = Campaign(
        candidate_data=candidate_data,
        agent=meta_agent, experiment=experiment,
        analyzer=None, s3_prefix=name, s3_bucket=bucket,
        create_seed=1
    )
    import nose; nose.tools.set_trace()
    loop.auto_loop(n_iterations, initialize=True)
