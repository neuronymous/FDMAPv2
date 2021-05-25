from networkx import MultiDiGraph, set_node_attributes

from DecPOMDPSimulator.PolicyGraphFormatter import SARSOPPolicyGraphFormatter


class NodeData(dict):
    def get_action(self):
        return self['action']

    def get_mls(self):
        return self['mls']

    def get_mls_prob(self):
        return self['mls_prob']


class EdgeData(dict):
    def get_obs(self):
        return self['obs']

    def get_obs_prob(self):
        return self['obs_prob']


class SARSOPGraph:
    def __init__(self, graph):
        self.formatter = SARSOPPolicyGraphFormatter
        self.graph = graph
        for node_id in graph.nodes():
            graph.nodes[node_id]['label'] = graph.nodes[node_id]['label'].replace('\\n', '')
        for edge in graph.edges():
            graph.adj[edge[0]][edge[1]][0]['label'] = graph.adj[edge[0]][edge[1]][0]['label'].replace('\\n', '')

    def get_action_from_node(self, node_id):
        return self.formatter.extract_action_name_from_node(self.get_node(node_id))

    def get_obs_from_edge(self, u, v):
        return self.formatter.extract_obs_name_from_edge(self.get_edge(u, v))

    def get_obs_prob_from_edge(self, u, v):
        return self.formatter.extract_obs_prob_from_edge(self.get_edge(u, v))

    def get_edge(self, u, v):
        return self.graph.adj[u][v][0]

    def add_edge(self, u, v, edge_data=None):
        self.graph.add_edge(u, v, **edge_data)

    def get_node(self, node_id):
        return self.graph.nodes()[node_id]

    def add_node(self, node_id, node_data=None):
        if node_data is not None:
            self.graph.add_node(node_id, **node_data)

    def remove_edge(self, *args, **kwargs):
        return self.graph.remove_edge(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        return self.graph.remove_node(*args, **kwargs)

    def has_node(self, *args, **kwargs):
        return self.graph.has_node(*args, **kwargs)

    def predecessors(self, *args, **kwargs):
        return self.graph.predecessors(*args, **kwargs)

    def nodes(self, *args, **kwargs):
        return self.graph.nodes(*args, **kwargs)

    def neighbors(self, *args, **kwargs):
        return self.graph.neighbors(*args, **kwargs)

    def copy(self):
        return SARSOPGraph(self.graph.copy())

    def in_edges(self, *args, **kwargs):
        return self.graph.in_edges(*args, *kwargs)

    def num_obs_vars(self):
        return len(
            list(list(self.graph.adj.items())[0][1].items())[0][1][0]['label'].split('(')[1].split(')')[0].split(','))
