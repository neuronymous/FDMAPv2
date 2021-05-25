from GraphAlignment.PoliciesAlignment import AlignmentAlgorithm, Node, Edge


def remove_private_foreign_nodes(graph, is_private_foreign_node, num_obs_vars=None, *args):
    foreign_nodes = []
    for node_id in graph.nodes():
        if is_private_foreign_node(node_id):
            foreign_nodes.append(node_id)
    root_noop_index = 0
    for node_id in foreign_nodes:
        predecessors = list(graph.predecessors(node_id)).copy()
        neighbors = list(graph.neighbors(node_id)).copy()
        for neighbor_id in neighbors:
            if len(predecessors) == 0:
                noop_node_id = node_id + "_root_noop_%d" % root_noop_index
                root_noop_index += 1
                noop_node_instance = Node(graph.get_node(node_id))
                noop_node_instance.action = 'action_idle'
                noop_node_data = noop_node_instance.generate_networkx_node_data()
                noop_edge_data = Edge.generate_empty_obs_edge_data(num_obs_vars, 'null')
                graph.add_node(noop_node_id, node_data=noop_node_data)
                graph.add_edge(noop_node_id, node_id, edge_data=noop_edge_data)
                predecessors.append(noop_node_id)
            for parent_id in predecessors:
                edge_data = graph.get_edge(parent_id, node_id)
                graph.remove_edge(parent_id, node_id)
                graph.add_edge(parent_id, neighbor_id, edge_data=edge_data)
            graph.remove_edge(node_id, neighbor_id)
        graph.remove_node(node_id)
    return graph


def remove_foreign_nodes_v1(graph, is_foreign_node, *args):
    """All foreign nodes are removed.
    """
    foreign_nodes = []
    for node_id in graph.nodes():
        if is_foreign_node(node_id):
            foreign_nodes.append(node_id)

    for node_id in foreign_nodes:
        predecessors = list(graph.predecessors(node_id)).copy()
        neighbors = list(graph.neighbors(node_id)).copy()
        if len(neighbors) > 1:
            raise Exception("A foreign node must have only one neighbor")
        for neighbor_id in neighbors:
            for parent_id in predecessors:
                edge_data = graph.get_edge(parent_id, node_id)
                graph.remove_edge(parent_id, node_id)
                graph.add_edge(parent_id, neighbor_id, edge_data=edge_data)
            graph.remove_edge(node_id, neighbor_id)
        graph.remove_node(node_id)
    return graph


def remove_foreign_nodes_v2(graph, is_foreign_node, noop_action):
    """Foreign nodes without loopback are turned idle
    rest are removed
    """
    foreign_nodes = []
    for node_id in graph.nodes():
        if is_foreign_node(node_id):
            foreign_nodes.append(node_id)

    for node_id in foreign_nodes:
        predecessors = list(graph.predecessors(node_id)).copy()
        neighbors = list(graph.neighbors(node_id)).copy()
        if len(neighbors) > 1:
            raise Exception("A foreign node must have only one neighbor")
        is_back_looped = True if len(neighbors) == 1 and neighbors[0] in predecessors else False
        if is_back_looped:
            for neighbor_id in neighbors:
                for parent_id in predecessors:
                    edge_data = graph.get_edge(parent_id, node_id)
                    graph.remove_edge(parent_id, node_id)
                    graph.add_edge(parent_id, neighbor_id, edge_data=edge_data)
                graph.remove_edge(node_id, neighbor_id)
            graph.remove_node(node_id)
        else:
            noop_data = AlignmentAlgorithm._create_preceding_noop_node_data(node_data=graph.get_node(node_id),
                                                                            noop_action=noop_action)
            graph.nodes()._nodes[node_id] = noop_data
    return graph


def remove_foreign_nodes_v3(graph, is_foreign_node, noop_action):
    """Foreign nodes without loopback are turned idle
    rest are removed, edges are looped on neighbors
    """
    foreign_nodes = []
    for node_id in graph.nodes():
        if is_foreign_node(node_id):
            foreign_nodes.append(node_id)

    for node_id in foreign_nodes:
        predecessors = list(graph.predecessors(node_id)).copy()
        neighbors = list(graph.neighbors(node_id)).copy()
        if len(neighbors) > 1:
            raise Exception("A foreign node must have only one neighbor")

        noop_data = AlignmentAlgorithm._create_preceding_noop_node_data(node_data=graph.get_node(node_id),
                                                                        noop_action=noop_action)
        graph.nodes()._nodes[node_id] = noop_data
        is_back_looped = True if len(neighbors) == 1 and neighbors[0] in predecessors else False
        if is_back_looped:
            for neighbor_id in neighbors:
                if neighbor_id in predecessors:
                    edge_data = graph.get_edge(neighbor_id, node_id)
                    graph.remove_edge(neighbor_id, node_id)
                    graph.add_edge(neighbor_id, neighbor_id, edge_data=edge_data)

    return graph


def repeat_collaborative_actions(graph, is_collaborative_action, num_repeats):
    if num_repeats <= 0:
        return
    for node_id in list(graph.nodes()).copy():  # we add on the fly
        action = graph.get_action_from_node(node_id)
        node_data = graph.get_node(node_id)
        if is_collaborative_action(action):
            neighbors = list(graph.neighbors(node_id))
            if len(neighbors) != 1:
                raise Exception("A collaborative node must have only one neighbor")
            neighbor_id = neighbors[0]

            new_nodes_ids = [node_id + "_repeat_%d" % (i + 1) for i in range(num_repeats)]
            edge_data = graph.get_edge(node_id, neighbor_id)
            graph.remove_edge(node_id, neighbor_id)

            for idx, new_node_id in enumerate(new_nodes_ids):
                graph.add_node(new_node_id, node_data.copy())
                if idx == 0:
                    graph.add_edge(node_id, new_node_id, edge_data=edge_data.copy())
                if idx == (num_repeats - 1):
                    graph.add_edge(new_node_id, neighbor_id, edge_data=edge_data.copy())
                if idx > 0:
                    graph.add_edge(new_nodes_ids[idx - 1], new_node_id, edge_data=edge_data.copy())

    return graph

# def project_by_noop(graph, agent, problem_constants):
#     def is_foreign_node(node_id):
#         action_name = graph.get_action_from_node(node_id)
#         return action_name != problem_constants.idle_action() and \
#                agent not in problem_constants.extract_agents_from_action(action_name)
#
#     foreign_nodes = []
#     for node_id in graph.nodes():
#         if is_foreign_node(node_id):
#             foreign_nodes.append(node_id)
#
#     seen_nodes = []
#     for node_id in foreign_nodes:
#         seen_nodes.append(node_id)
#         predecessors = list(graph.predecessors(node_id)).copy()
#         neighbors = list(graph.neighbors(node_id)).copy()
#         noop_data = AlignmentAlgorithm._create_preceding_noop_node_data(node_data=graph.get_node(node_id),
#                                                                         noop_action=problem_constants.idle_action())
#         graph.nodes()._nodes[node_id] = noop_data
#         for seen_node_id in seen_nodes:
#             if seen_node_id in neighbors:
#                 graph.remove_edge(node_id, seen_node_id)
#
#
#
#     return graph
