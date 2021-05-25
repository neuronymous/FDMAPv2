from abc import ABC, abstractmethod
from collections.abc import Sequence
import networkx
import pydot

from DecPOMDPSimulator.PolicyGraphFormatter import FormatterFactory


class Policy(ABC):
    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update_by_obs(self, obs):
        pass

    @abstractmethod
    def reset(self):
        pass


class NodeData:
    def __init__(self, action):
        self.action = action


class EdgeData:
    def __init__(self, obs):
        self.obs = obs


class PolicyGraph(Policy):
    def __init__(self, dot_path, graph_format):
        self.graph = networkx.nx_pydot.from_pydot(pydot.graph_from_dot_file(dot_path)[0])
        self.formatter = FormatterFactory.create_formatter(graph_format)
        root_node_id = self.find_root_id(self.graph)
        if root_node_id is None:
            if self.graph.has_node(self.formatter.get_root_id()):
                root_node_id = self.formatter.get_root_id()
            else:
                raise Exception("Graph root cannot be determined")

        self.cur_node_id = root_node_id
        for edge in self.graph.edges():
            edge_data = self.graph.adj[edge[0]][edge[1]]
            edge_obs = self.formatter.extract_obs_name_from_edge(edge_data)
            edge_data['0']['data'] = EdgeData(edge_obs)

        for node in self.graph.nodes():
            node_data = self.graph.nodes()[node]
            node_action = self.formatter.extract_action_name_from_node(node_data)
            node_data['data'] = NodeData(node_action)

    def get_action(self, node_id=None):
        if node_id is None:
            node_id = self.cur_node_id
        return self.graph.nodes()[node_id]['data'].action

    def set_action(self, action, node_id=None):
        if node_id is None:
            node_id = self.cur_node_id
        self.graph.nodes()[node_id]['data'].action = action

    def update_by_obs(self, obs):
        obs = self._normalize_obs(obs)
        branch_made = False
        for neighbor in list(self.graph.successors(self.cur_node_id)):
            if self.graph.adj[self.cur_node_id][neighbor]['0']['data'].obs == obs:
                branch_made = True
                self.cur_node_id = neighbor
                break

        if not branch_made:
            print("No branch matched this observation")

    @staticmethod
    def _normalize_obs(obs):
        if not isinstance(obs, str) and isinstance(obs, Sequence):
            return ','.join(obs)
        return obs

    def reset(self):
        self.cur_node_id = self.formatter.get_root_id() if self.graph.has_node(self.formatter.get_root_id()) \
            else self.find_root_id(self.graph)

    def find_root_id(self, graph):
        for node_id in graph.nodes():
            if len(graph.in_edges(node_id)) == 0:
                return node_id
        return None  # TODO we're screwed


def replace_matching_actions(policy_graph, action_match_function, new_action):
    for node_id in policy_graph.graph.nodes():
        if action_match_function(policy_graph.get_action(node_id)):
            policy_graph.set_action(action=new_action, node_id=node_id)
