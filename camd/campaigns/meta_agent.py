#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
"""
This module provides resources for agent optimization campaigns
"""
from taburu.table import ParameterTable
from camd import CAMD_S3_BUCKET
from camd.agent.meta import AGENT_PARAMS, \
    convert_parameter_table_to_dataframe
from camd.campaigns.base import Campaign
import pickle
import boto3
import botocore

META_AGENT_PREFIX = "meta_agent"


class MetaAgentCampaign(Campaign):
    @staticmethod
    def reserve(name, experiment, analyzer, agent_pool=None,
                bucket=CAMD_S3_BUCKET):
        """
        Method for making an reservation for a given campaign in s3

        Args:
            name (str): reservation name
            experiment (Experiment, DataFrame): complete
                experiment provisioned for agent testing,
                typically a LocalAgentSimulation
            analyzer (camd.analysis.Analyzer): analyzer for which
                to analyze/judge meta-agent campaigns
            agent_pool (ParameterTable): parameter table corresponding
                to agents to be tested
            bucket (str): name of s3 bucket

        Returns:
            (None)

        """
        # Check reservation
        client = boto3.client("s3")
        prefix = "{}/{}".format(META_AGENT_PREFIX, name)
        result = client.list_objects(Bucket=bucket, Prefix=prefix)
        if result.get('Contents'):
            raise ValueError(
                "{} exists in s3, please choose another name".format(name))

        # Store agent pool
        agent_pool = agent_pool or ParameterTable(AGENT_PARAMS)
        pickled_agent = pickle.dumps(agent_pool)
        client.put_object(
            Bucket=bucket,
            Key="{}/agent_pool.pickle".format(prefix),
            Body=pickled_agent
        )

        # Store experiment
        pickled_experiment = pickle.dumps(experiment)
        client.put_object(
            Bucket=bucket,
            Key="{}/experiment.pickle".format(prefix),
            Body=pickled_experiment
        )

        # Store analyzer
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
        prefix = "{}/{}".format(META_AGENT_PREFIX, name)
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
        prefix = "{}/{}".format(META_AGENT_PREFIX, name)
        all_objs = []
        for obj_name in ['agent_pool', 'experiment', 'analyzer']:
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
    def from_reserved_name(cls, name, meta_agent=None, bucket=CAMD_S3_BUCKET):
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
        agent_pool, experiment, analyzer = cls.load_pickled_objects(name, bucket)
        s3_prefix = "{}/{}".format(META_AGENT_PREFIX, name)
        candidate_data = convert_parameter_table_to_dataframe(agent_pool)
        return cls(
            candidate_data=candidate_data,
            agent=meta_agent, experiment=experiment,
            analyzer=analyzer, s3_prefix=s3_prefix, s3_bucket=bucket,
            create_seed=1
        )

    def autorun(self):
        """
        Convenience method for standard running procedure

        Returns:
            (None)

        """
        self.auto_loop(n_iterations=5, initialize=True)


class StabilityMetaAgentCampaign(MetaAgentCampaign):
    """
    Convenience class to construct MetaAgent
    campaigns around stability
    """
    pass


# TODO: move this into analysis
from camd.analysis import AnalyzerBase


# For now this is specific to stability
class StabilityCampaignAnalyzer(AnalyzerBase):
    def __init__(self):
        pass

    def analyze(self, new_experimental_results, seed_data):
        new_experimental_results.total_10 = None
        new_experimental_results.total_25 = None
        new_experimental_results.total_50 = None
        for key, row in new_experimental_results.iterrows():
            history = row.campaign.history
            new_experimental_results.loc[key, "discovered_10"] = history.loc[10, 'total_stable']
            new_experimental_results.loc[key, "discovered_25"] = history.loc[25, 'total_stable']
            new_experimental_results.loc[key, "discovered_51"] = history.iloc[-1]['total_stable']
        seed_data = seed_data.append(new_experimental_results)
        summary = new_experimental_results.loc["discovered_10": "discovered_51"]
        return summary, seed_data
