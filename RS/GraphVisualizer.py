import graphviz
import os, shutil
import pydot

GRAPHSDIR = "../Resources/graphs/"
COLOREDGRAPHSDIR = os.path.join(GRAPHSDIR, 'colored/')
GRAPHSIMGSDIR = os.path.join(GRAPHSDIR, 'images/')
SRCFORMAT = 'dot'
DSTFORMAT = 'png'

GRAPHS_TO_CONVERT = None
GRAPHS_TO_COLOR = None

AGENT_TO_COL = {
    'car1': 'coral',
    'car2': 'cadetblue1',
}

ACTION_PREFIX_TO_COL = {
    'action_idle': 'grey'
}


def convert_dots_to_png(colored=True):
    graphs_dir = COLOREDGRAPHSDIR if colored else GRAPHSDIR
    graphs = GRAPHS_TO_CONVERT if GRAPHS_TO_CONVERT is not None else os.listdir(graphs_dir)
    for f_name in graphs:
        if 'RS' in f_name:
            f_path = os.path.join(graphs_dir, f_name)
            img_name = f_name + '.png'
            if f_name.endswith(SRCFORMAT):
                graphviz.render(SRCFORMAT, DSTFORMAT, filepath=f_path)
                shutil.move(os.path.join(graphs_dir, img_name), os.path.join(GRAPHSIMGSDIR, img_name))


class Node:
    action_token = 'A'
    mls_token = 'Y'
    split_label = 'l'

    def __init__(self, dot_node):
        label = dot_node.get_attributes()['label']
        comps = label.replace('"', '').split(
            '\\%s' % Node.split_label)

        for comp in comps:
            if comp.startswith(Node.action_token):
                self.action = comp.replace(Node.action_token, '').strip()

            if comp.startswith(Node.mls_token):
                [self.most_likely_state, self.mls_prob] = comp.replace(Node.mls_token, '') \
                    .strip().split(' ')
                self.mls_prob = float(self.mls_prob)

    def generate_label(self):
        return "%s %s %s\\%s%s %s" % (
            self.action_token, self.most_likely_state, str(self.mls_prob), self.split_label, self.action_token,
            self.action)


def color_graph_by_agent(graph_path, output_path, agent_to_col=None, action_prefix_to_col=None):
    g = pydot.graph_from_dot_file(graph_path)
    for node in g[0].get_nodes():
        node_attributes = node.get_attributes()
        node_agent = None
        node_instance = Node(node)
        node_action = node_instance.action.strip('()')  # TODO change to ALignemnt's node class

        if agent_to_col is not None:
            for agent in agent_to_col:
                if agent in node_action:
                    node_agent = agent
            if node_agent is not None:
                node_attributes['color'] = 'black'
                node_attributes['fillcolor'] = agent_to_col[node_agent]
                node_attributes['style'] = "filled"

        if action_prefix_to_col is not None:
            for action_prefix, color in action_prefix_to_col.items():
                if node_action.startswith(action_prefix):
                    node_attributes['color'] = 'black'
                    node_attributes['fillcolor'] = color
                    node_attributes['style'] = "filled"

    print(output_path)
    g[0].write(output_path, format='raw')


if __name__ == "__main__":
    graphs = GRAPHS_TO_COLOR if GRAPHS_TO_COLOR is not None else os.listdir(GRAPHSDIR)
    graphs = list(filter(lambda x: os.path.isfile(os.path.join(GRAPHSDIR, x)) and 'RS' in x, graphs))
    for graph in graphs:
        input_path = os.path.join(GRAPHSDIR, graph)
        output_path = os.path.join(COLOREDGRAPHSDIR, os.path.basename(input_path))
        output_path = output_path[:-4] + "_COLOR.dot"
        color_graph_by_agent(input_path, output_path, AGENT_TO_COL, ACTION_PREFIX_TO_COL)

    convert_dots_to_png(colored=True)
