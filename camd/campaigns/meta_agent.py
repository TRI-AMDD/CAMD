#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.agent.meta import AGENT_PARAMS, \
    convert_parameter_table_to_dataframe
from camd.agent.base import RandomAgent
from camd.experiment.atf import LocalAgentSimulation
from camd.loop import Campaign
import pickle
import boto3
import botocore


class MetaAgentCampaign(Campaign):
    @staticmethod
    def reserve(name, dataframe, analyzer,
                agent_pool=None, bucket=CAMD_S3_BUCKET):
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
        pickled_analyzer = pickle.dumps(analyzer)
        client.put_object(
            Bucket=bucket,
            Key="{}/analyzer.pickle".format(prefix),
            Body=pickled_analyzer
        )

    @staticmethod
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
        agent_pool = MetaAgentCampaign.load_pickled_objects(name, bucket)[0]
        agent_pool.extend(params)

        pickled_agent = pickle.dumps(agent_pool)
        client.put_object(
            Bucket=bucket,
            Key="{}/agent_pool.pickle".format(prefix),
            Body=pickled_agent
        )

    @staticmethod
    def load_pickled_objects(name, bucket=CAMD_S3_BUCKET):
        """
        Loads pickled objects for given named campaign
        and bucket

        Args:
            name (str): name of campaign
            bucket (str): name of bucket

        Returns:
            (ParameterTable): parameter table of agents
            (DataFrame): dataframe for campaign
            (Analyzer): analyzer for campaign simulated
                experiments

        """
        client = boto3.client("s3")
        prefix = "agent_testing/{}".format(name)
        all_objs = []
        for obj_name in ['agent_pool', 'atf_data', 'analyzer']:
            try:
                raw_obj = client.get_object(
                    Bucket=bucket,
                    Key="{}/{}.pickle".format(prefix, obj_name),
                )['Body']
            except botocore.exceptions.ClientError as e:
                raise ValueError(
                    "{} does not exist in s3, cannot update agent pool".format(name)
                )
            obj = pickle.loads(raw_obj.read())
            all_objs.append(obj)
        return all_objs

    @classmethod
    def from_name(cls, name, meta_agent=None, bucket=CAMD_S3_BUCKET):
        """
        Invokes a MetaAgent Campaign from a reserved name

        Args:
            name (str): name of the campaign to be run
            meta_agent (HypothesisAgent): meta-agent with
                which to select the agent simulations to
                be run
            bucket (str): name of the bucket from which
                to invoke the campaign

        Returns:
            (MetaAgentCampaign) - meta-agent campaign corresponding
                to the campaign

        """
        agent_pool, atf_data, analyzer = cls.load_pickled_objects(name, bucket)
        meta_agent = meta_agent or RandomAgent(n_query=1)
        experiment = LocalAgentSimulation(
            atf_dataframe=atf_data, analyzer=analyzer,
            iterations=50, n_seed=1)
        candidate_data = convert_parameter_table_to_dataframe(agent_pool)
        return cls(
            candidate_data=candidate_data,
            agent=meta_agent, experiment=experiment,
            analyzer=analyzer, s3_prefix=name, s3_bucket=bucket,
            create_seed=1
        )

    def autorun(self):
        self.auto_loop(n_iterations=5)


from camd.analysis import AnalyzerBase


class CampaignAnalyzer(AnalyzerBase):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def analyze(self, seed_data, new_experimental_results):
        import nose; nose.tools.set_trace()
        for key, row in new_experimental_results.iterrows():
            pass

