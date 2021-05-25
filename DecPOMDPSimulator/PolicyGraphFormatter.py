from abc import ABC, abstractmethod


class FormatterFactory:

    @staticmethod
    def create_formatter(graph_format):
        if graph_format == "SARSOP":
            return SARSOPPolicyGraphFormatter


class Formatter(ABC):
    """
    Receives node_data and edge_data that were constructed using pydot -> networkx graph
    """

    @classmethod
    @abstractmethod
    def extract_action_name_from_node(cls, node_data):
        pass

    @classmethod
    @abstractmethod
    def extract_obs_name_from_edge(cls, edge_data):
        pass

    @staticmethod
    @abstractmethod
    def get_root_id():
        pass


class SARSOPPolicyGraphFormatter(Formatter):
    action_token = 'A'
    mls_token = 'Y'
    split_token = '\\l'
    observation_token = 'o'
    root_id = 'root'

    @classmethod
    def extract_action_name_from_node(cls, node_data):
        label = node_data['label']
        comps = label.replace('"', '').split(cls.split_token)

        for comp in comps:
            if comp.startswith(cls.action_token):
                action = comp.replace(cls.action_token, '').strip('() ')
                return action

    @classmethod
    def extract_obs_name_from_edge(cls, edge_data):
        label = edge_data['0']['label']
        comps = label.replace('"', '').split(cls.split_token)

        for comp in comps:
            if comp.startswith(cls.observation_token):
                obs = comp.split(' ')[1].strip('() ')
                return obs

    @staticmethod
    def get_root_id():
        return SARSOPPolicyGraphFormatter.root_id

    @classmethod
    def extract_obs_prob_from_edge(cls, edge_data):
        label = edge_data[0]['label']
        comps = label.replace('"', '').split(cls.split_token)

        for comp in comps:
            if comp.startswith(cls.observation_token):
                obs_prob = float(comp.split(' ')[2].strip('() '))
                return obs_prob

    @classmethod
    def extract_mls_prob_from_node(cls, node_data):
        label = node_data['label']
        comps = label.replace('"', '').split(cls.split_token)

        for comp in comps:
            if comp.startswith(cls.mls_token):
                mls_prob = float(comp.replace(cls.mls_token, '').strip('() ').split(' ')[1])
                return mls_prob

    @classmethod
    def extract_mls_from_node(cls, node_data):
        label = node_data['label']
        comps = label.replace('"', '').split(cls.split_token)

        for comp in comps:
            if comp.startswith(cls.mls_token):
                mls = comp.replace(cls.mls_token, '').strip('() ').split(' ')[0]
                return mls
