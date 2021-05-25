from queue import Queue

import networkx
import pydot

from DecPOMDPSimulator.PolicyGraphFormatter import SARSOPPolicyGraphFormatter
from GraphAlignment.GraphWrappers import SARSOPGraph

DEFAULT_ROOT_ID = SARSOPPolicyGraphFormatter.root_id


class BFS:
    @staticmethod
    def graph_from_dotpath(dotpath):
        return SARSOPGraph(networkx.nx_pydot.from_pydot(pydot.graph_from_dot_file(dotpath)[0]))

    def __init__(self, graph):
        self.graph = graph
        self.queue = Queue()
        self.visited = {node: False for node in graph.nodes()}
        self.has_backloop = {node: False for node in graph.nodes()}  # Backloop - a back edge to a higher predecessor
        self.depths = {node: 0 for node in graph.nodes()}

        self.root_id = self.find_root_node()
        self.queue.put(self.root_id)
        self.visited[self.root_id] = True
        self.depths[self.root_id] = 1
        self.expander = {node_id: None for node_id in graph.nodes()}

    def expand_node(self, avoid_neighbors_insertion=False):
        if not self.queue.empty():
            cur_node = self.queue.get()
            neighbors = self.graph.neighbors(cur_node)
            loopback_forming_parents = []
            inserted_neighbors = []
            for neighbor in neighbors:
                if not self.visited[neighbor]:
                    if avoid_neighbors_insertion:
                        inserted_neighbors.append(neighbor)
                    else:
                        self.queue.put(neighbor)
                        self.visited[neighbor] = True
                        self.depths[neighbor] = self.depths[cur_node] + 1
                        self.expander[neighbor] = cur_node

                # If neighbor is visited, check for loopback (can be non-parent, and same leveled)
                elif self.is_real_parent_of_node(neighbor, cur_node):
                    self.has_backloop[cur_node] = True
                    loopback_forming_parents.append(neighbor)
            if avoid_neighbors_insertion:
                return cur_node, inserted_neighbors
            return cur_node
        return None

    def manually_visit_node(self, parent_id, node_id):
        self.queue.put(node_id)
        self.visited[node_id] = True
        self.depths[node_id] = self.depths[parent_id] + 1
        self.expander[node_id] = parent_id

    def expand_until_end(self):
        while self.expand_node() is not None:
            print("Node expanded")

    def _get_all_paths_to_root(self, node_id, encountered_nodes, visited_nodes_only):
        """Returns all paths to root
        Can handle cycles by providing the encountered nodes (notice the copy!)
        Memory issues might happen for very large graphs of course
        visited_nodes_only ensures paths only consist nodes that were visited through the BFS"""
        result = []
        valid_predecessors = []
        if encountered_nodes is None:
            encountered_nodes = set()
        encountered_nodes.add(node_id)

        if self.is_root_node(node_id):  # Break condition
            return [(self.root_id,)]

        for parent_id in self.graph.predecessors(node_id):
            if ((not visited_nodes_only) or self.visited[parent_id]) \
                    and parent_id not in encountered_nodes:
                valid_predecessors.append(parent_id)

        for predecessor_id in valid_predecessors:
            for path in self._get_all_paths_to_root(predecessor_id, encountered_nodes.copy(), visited_nodes_only):
                result.append((node_id, self.graph.get_edge(predecessor_id, node_id), *path))

        return result

    def get_all_visited_paths_to_root(self, node_id):
        return self._get_all_paths_to_root(node_id=node_id, encountered_nodes=None, visited_nodes_only=True)

    def get_all_simple_paths_to_root(self, node_id):
        return self._get_all_paths_to_root(node_id=node_id, encountered_nodes=None, visited_nodes_only=False)

    def is_root_node(self, node):
        return node == self.root_id

    def find_root_node(self):
        if DEFAULT_ROOT_ID in self.graph.nodes():
            return DEFAULT_ROOT_ID
        for node_id in self.graph.nodes():
            if len(self.graph.in_edges(node_id)) == 0:
                return node_id
        return None  # Should not get here, no root found

    def get_expanders(self, cur_node):
        pass

    def has_finished(self):
        return self.queue.empty()

    def is_real_parent_of_node(self, parent_id, node_id):
        return self.visited[parent_id] and \
               self.depths[parent_id] < self.depths[node_id] and \
               parent_id in self.graph.predecessors(node_id)
