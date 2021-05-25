from collections import namedtuple
import pydot
import sys, os
from graphviz import Source

TREE_FILE = './test-graph.dot'

NodeData = namedtuple('PolicyNode', 'action ml_state ml_state_prob')
EdgeData = namedtuple('PolicyEdge', 'obs obs_prob')


def parse_node_label(label):
    label = label.replace('\\l', ' ').replace('(', '').replace(')', '')
    label_components = label.split(' ')

    state = label_components[1]
    state_prob = float(label_components[2])
    action = label_components[4]

    return NodeData(action=action, ml_state=state, ml_state_prob=state_prob)


def parse_edge_label(label):
    label = label.replace('\\l', ' ').replace('(', '').replace(')', '')
    label_components = label.split(' ')

    obs = label_components[1]
    obs_prob = float(label_components[2])

    return EdgeData(obs=obs, obs_prob=obs_prob)


def tree_from_dot_file(trace_file):
    tree = pydot.graph_from_dot_file(trace_file)[0]
    nodes = tree.get_node_list()
    edges = tree.get_edge_list()

    for node in nodes:
        node_attr = node.get_attributes()
        node_attr['data'] = parse_node_label(node_attr['label'])

    for edge in edges:
        edge_attr = edge.get_attributes()
        edge_attr['data'] = parse_edge_label(edge_attr['label'])

    return tree


if __name__ == "__main__":
    tree = tree_from_dot_file(TREE_FILE)
    s = Source.from_file(TREE_FILE)
    s.view()


