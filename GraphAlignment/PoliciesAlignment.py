"""
Algorithm 1:
Based on single agent policies only
Given the target policy P, and the rest of the policies Q_1, ... Q_m
We initialize BFS instances on all graphs
We expand nodes in P's BFS, until we reach a public action. When reaching a public action, we want to find the proper
alignment for it. (we need to consider parallel public actions, perhaps same level public actions)
We start to expand on Q_1, ... Q_m, while stopping the expansion in each graph, when we either find the public action
or the queue is empty.
If the queue is empty we can safetly say that this agent doesn't require alignment for P's public
action.
Else we've found the action, and we stop.
When we stop on all graphs, we need to choose the maximal  chain required, using the maximal counter
among all found. Graphs we depleted yield an  requirement of 0 of course
Once we pick the  amount we go back to P, and add that amount as a IDLE chain before the public action.
If the public action has a loopback (for now assuming towards the direct parent),
we add the IDLE chain to its parent instead. We then continue to exapnd P, while keeping the already expanded BFS
among Q_1,...Q_m, and resetting those who were depleted.

Think about:
- loopback?
- we need same amount of noop to several public actions at the same level basically
- other ways to detect similar nodes that require same IDLE chain: keeping track of histories of public actions, or
look at context, at least most likely state

- For now, we consider A1 and A2 as matching actions in the tree, if they match on some chain of public actions that lead to them (ancestors chain)
perhaps we would prefer to make a more subtle matching by considering observations.
"""
from collections import Counter
from math import ceil
from random import randint
from DecPOMDPSimulator.POMDPXMarks import POMDPXMarks
from GraphAlignment.BFS import BFS


class Edge:
    obs_token = 'o'
    split_token = '\\l'

    @staticmethod
    def generate_empty_obs_edge_data(obs_size, null_obs):
        return {
            'label': f'\"{Edge.obs_token} ({",".join([null_obs for i in range(obs_size)])}) 1{Edge.split_token}\"'}


class Node:
    action_token = 'A'
    mls_token = 'Y'
    split_token = '\\l'

    def __init__(self, node_data):
        label = node_data['label']
        comps = label.replace('"', '').split(self.split_token)

        for comp in comps:
            if comp.startswith(Node.action_token):
                self.action = comp.replace(Node.action_token, '').strip('() ')

            if comp.startswith(Node.mls_token):
                [self.most_likely_state, self.mls_prob] = comp.replace(Node.mls_token, '') \
                    .strip().split(' ')

                self.most_likely_state = self.most_likely_state.strip('() ')
                self.mls_prob = float(self.mls_prob)

    def generate_networkx_node_data(self):
        return {'label': self.generate_label()}

    def generate_label(self):
        return f'\"{self.mls_token} ({self.most_likely_state}) {self.mls_prob}{self.split_token}{self.action_token} ({self.action}){self.split_token}\"'


class AlignmentAlgorithm:
    def __init__(self, graphs, problem_constants, num_obs_vars_per_agent, exact_match_id=False, share_public_actions=False,
                 consider_preceding_noops=True):
        self.graphs = graphs
        self.get_id_func = self._id_extractor_public_actions
        self.is_public_action = problem_constants.is_public_action
        self.is_agent_in_action = lambda agent, action: agent in problem_constants.extract_agents_from_action(action)
        self.noop_action = problem_constants.idle_action()
        self.null_obs = problem_constants.NULL_OBS
        self.exact_match_id = exact_match_id
        self.agents_public_actions = [[graph.get_action_from_node(node) for node in graph.nodes() if
                                       self.is_public_action(graph.get_action_from_node(node))] for graph in graphs]
        self.current_num_obs_vars = None
        self.num_obs_vars_per_agent = num_obs_vars_per_agent
        self.share_public_actions = share_public_actions
        self.consider_preceding_noops = consider_preceding_noops

    def calculate_aligned_policy(self, agent_index, agent_symbol):
        main_graph = self.graphs[agent_index]
        self.current_num_obs_vars = self.num_obs_vars_per_agent[agent_index]
        main_bfs = BFS(main_graph)
        node_to_noops_reqs = {}
        node_to_preceding_noops = Counter()
        while not main_bfs.has_finished():
            cur_node_id = main_bfs.expand_node()

            # We decrease the number of preceding noops
            possible_preceding_noops = [node_to_preceding_noops[parent] for parent in
                                        main_graph.predecessors(cur_node_id)]
            # Consider the branch with the least noops, we need the noop req to be satisfied in the max branch
            min_preceding_noops = min(possible_preceding_noops) if len(possible_preceding_noops) > 0 else 0
            node_to_preceding_noops[cur_node_id] = min_preceding_noops

            cur_node_action = main_graph.get_action_from_node(cur_node_id)
            if self.is_public_action(cur_node_action) and \
                    (self.share_public_actions or self.is_agent_in_action(agent_symbol, cur_node_action)):

                # Each path forms a different identifier, we max our noop requirement to match all
                identifiers = [self.get_id_func(path, main_graph) for path in
                               main_bfs.get_all_visited_paths_to_root(cur_node_id)]
                identifiers_noop_requirements = [0 for _ in identifiers]

                for idx, identifier in enumerate(identifiers):
                    identifiers_noop_requirements[idx] = self._calculate_identifier_noops_requirement(agent_index,
                                                                                                      identifier)
                total_noops_required = max(identifiers_noop_requirements)
                compensation_term = main_bfs.depths[cur_node_id] - 1
                if self.consider_preceding_noops:
                    compensation_term += min_preceding_noops

                if total_noops_required - compensation_term > 0:
                    # we take the absolute noops requirement, and deduct the preceding noops and the depth - 1
                    node_to_noops_reqs[cur_node_id] = total_noops_required - compensation_term
                    node_to_preceding_noops[cur_node_id] += node_to_noops_reqs[cur_node_id]

        return self._construct_nooped_graph(node_to_noops_reqs, agent_index, main_bfs)

    def _construct_nooped_graph(self, node_to_noops, agent_index, bfs):
        res_graph = self.graphs[agent_index].copy()
        for node, noops_num in node_to_noops.items():
            nooped_node = node
            parents = list(res_graph.predecessors(nooped_node))

            # Special case for single parent and loopback, switching nooped node to parent
            if bfs.has_backloop[node] and len(parents) == 1:
                nooped_node = parents[0]
                parents = list(res_graph.predecessors(nooped_node))

            parents_to_avoid = []

            # We avoid parents with lower depth, since we don't want to detach from them
            for parent in parents:
                if not bfs.is_real_parent_of_node(parent, nooped_node):
                    parents_to_avoid.append(parent)

            for i in range(noops_num):
                self._attach_noop_to_node(res_graph, nooped_node, parents_to_avoid, noop_index=i)
        return res_graph

    def _calculate_identifier_noops_requirement(self, agent_index, identifier):
        coagents_bfs = [BFS(graph) for idx, graph in enumerate(self.graphs) if idx != agent_index]
        finish_flags = [False for _ in coagents_bfs]
        noops_reqs = [0 for _ in coagents_bfs]
        while not all(finish_flags):
            running_indices = [i for i in range(len(finish_flags)) if not finish_flags[i]]
            for i in running_indices:
                cur_bfs = coagents_bfs[i]
                if not cur_bfs.has_finished():
                    cur_node, neighbors_to_insert = cur_bfs.expand_node(avoid_neighbors_insertion=True)
                    matching_path_len = int(
                        self._find_max_matching_path_len(identifier, cur_bfs, cur_node, agent_index))
                    if matching_path_len > 0:  # if match, we stem the expansion
                        noops_reqs[i] = max(noops_reqs[i], matching_path_len - 1)  # path_len includes the node itself
                    else:  # if no match found, perform visitation regularly
                        for neighbor in neighbors_to_insert:
                            cur_bfs.manually_visit_node(parent_id=cur_node, node_id=neighbor)
                else:
                    finish_flags[i] = True
        return max(noops_reqs)

    def _attach_noop_to_node(self, graph, node_id, parents_to_avoid, noop_index=None):
        predecessors = list(graph.predecessors(node_id)).copy()
        if parents_to_avoid is not None:
            for bad_parent in parents_to_avoid:
                predecessors.remove(bad_parent)

        if noop_index is None:
            noop_index = randint()
        noop_node_id = node_id + "_noop_%d" % noop_index
        noop_node_data = self._create_preceding_noop_node_data(graph.get_node(node_id))
        graph.add_node(noop_node_id, node_data=noop_node_data)

        for parent_id in predecessors:
            edge_data = graph.get_edge(parent_id, node_id)  # No duplicate edges
            graph.remove_edge(parent_id, node_id)
            graph.add_edge(parent_id, noop_node_id, edge_data=edge_data)  # retain the original edge data

        noop_edge_data = self._create_preceding_noop_edge_data()
        graph.add_edge(noop_node_id, node_id, edge_data=noop_edge_data)

    def _find_matching_path(self, identifier, bfs, node, identifier_agent_index):
        """Check if identifier is contained in some path to root"""
        # We're only interested in paths which match in their first node with the identifier
        if identifier[0] != bfs.graph.get_action_from_node(node):  # TODO generic, remove action concept
            return 0
        all_paths_to_root = bfs.get_all_simple_paths_to_root(node)
        for path in all_paths_to_root:
            if self.exact_match_id:
                path_matches = self._path_matches_identifier(identifier, bfs, path,
                                                             self.agents_public_actions[identifier_agent_index])
            else:
                path_matches = self._path_contains_identifier(identifier, bfs, path)
            if path_matches:
                return ceil(len(path) / 2.0)
        return 0

    def _find_max_matching_path_len(self, identifier, bfs, node, identifier_agent_index):
        """Check if identifier is contained in some path to root"""
        # We're only interested in paths which match in their first node with the identifier
        if identifier[0] != bfs.graph.get_action_from_node(node):  # TODO generic, remove action concept
            return 0
        all_paths_to_root = bfs.get_all_simple_paths_to_root(node)
        longest_matching_path_len = 0
        for path in all_paths_to_root:
            if self.exact_match_id:
                path_matches = self._path_matches_identifier(identifier, bfs, path,
                                                             self.agents_public_actions[identifier_agent_index])
            else:
                path_matches = self._path_contains_identifier(identifier, bfs, path)

            if path_matches:
                longest_matching_path_len = max(longest_matching_path_len, ceil(len(path) / 2.0))

        return longest_matching_path_len

    def _id_extractor_public_actions(self, path, graph):
        """Identifier is computed with respect to a path in the graph
        this ID calc concatenates the public actions along the path
        ID is always global with respect to the POMDPX
        """
        res = []
        for comp in path:
            if graph.has_node(comp):
                node_id = comp
                action = graph.get_action_from_node(node_id)
                if self.is_public_action(action):
                    res.append(action)
        return tuple(res)

    @staticmethod
    def _path_contains_identifier(identifier, bfs, path):
        id_len, path_len = len(identifier), len(path)
        if id_len > path_len:
            return False
        p2 = 0
        for p1 in range(path_len):
            if bfs.graph.has_node(path[p1]) and \
                    identifier[p2] == bfs.graph.get_action_from_node(path[p1]):
                p2 += 1
            if p2 == id_len:
                return True
        return False

    def _path_matches_identifier(self, identifier, bfs, path, identifier_domain_actions):
        projected_path = [node for node in path if
                          bfs.graph.has_node(node)
                          and self.is_public_action(bfs.graph.get_action_from_node(node))
                          and bfs.graph.get_action_from_node(node) in identifier_domain_actions]  # project the path
        id_len, path_len = len(identifier), len(projected_path)
        if id_len != path_len:
            return False
        p2 = 0
        for p1 in range(path_len):
            if bfs.graph.has_node(projected_path[p1]) and \
                    identifier[p2] == bfs.graph.get_action_from_node(projected_path[p1]):
                p2 += 1
            if p2 == id_len:
                return True
        return False

    def _create_preceding_noop_node_data(self=None, node_data=None, noop_action=None):
        node_instance = Node(node_data)
        node_instance.action = self.noop_action if self is not None else noop_action
        return node_instance.generate_networkx_node_data()

    def _create_preceding_noop_edge_data(self):
        return Edge.generate_empty_obs_edge_data(self.current_num_obs_vars, self.null_obs)
