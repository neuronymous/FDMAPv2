from time import time
from functools import partial
from itertools import product

import networkx
import sys

from numpy.compat import os_PathLike

from GraphAlignment import PostProcessUtils
from GraphAlignment.BFS import BFS
from GraphAlignment.PostProcessUtils import *
from GraphAlignment.PostProcessUtils import repeat_collaborative_actions
from POMDPX.POMDPXFactory import POMDPXProblemFactory
from conf import Config
from GraphAlignment.defaults import Defaults
import os


def create_aligned_solution(index_to_agent,
                            graph_paths,
                            exact_align,
                            post_version,
                            repeat_collab,
                            num_repeats, problem_name,
                            problem_constants,
                            agent_to_num_obsvars,
                            consider_preceding_noops=True,
                            share_public_actions=False,
                            num_align_rounds=1,
                            preprocess=False
                            ):
    result_paths = []

    def is_foreign_node_to_agent(agent_symbol, graph, node_id):
        action_name = graph.get_action_from_node(node_id)
        return action_name != problem_constants.idle_action() and \
               agent_symbol not in problem_constants.extract_agents_from_action(action_name)

    def private_foreign_node(agent_symbol, graph, node_id):
        action_name = graph.get_action_from_node(node_id)
        return is_foreign_node_to_agent(agent_symbol, graph, node_id) and not problem_constants.is_public_action(
            action_name)

    raw_graphs = [BFS.graph_from_dotpath(path) for path in graph_paths]

    if preprocess:
        graphs = []
        for idx, raw_graph in enumerate(raw_graphs):
            agent = index_to_agent[idx]

            is_private_foreign_node = partial(private_foreign_node, agent, raw_graph)
            graphs.append(PostProcessUtils.remove_private_foreign_nodes(raw_graph.copy(), is_private_foreign_node,
                                                                        raw_graph.num_obs_vars()))
            del raw_graph
    else:
        graphs = raw_graphs

    alignment_alg_params = {
        "exact_match_id": exact_align,
        "problem_constants": problem_constants,
        "num_obs_vars_per_agent": [graphs[i].num_obs_vars() for i in range(len(graphs))],
        "share_public_actions": share_public_actions,
        "consider_preceding_noops": consider_preceding_noops
    }

    # Align
    alg = AlignmentAlgorithm(graphs, **alignment_alg_params)

    for _ in range(num_align_rounds - 1):
        curr_aligned_graphs = [alg.calculate_aligned_policy(agent_index=idx, agent_symbol=agent)
                               for idx, agent in index_to_agent.items()]
        alg = AlignmentAlgorithm(curr_aligned_graphs, **alignment_alg_params)

    aligned_graphs = {}
    for idx, agent in index_to_agent.items():
        res = alg.calculate_aligned_policy(agent_index=idx, agent_symbol=agent)
        aligned_graphs[agent] = res

    # Post process
    processed_aligned_graphs = {}
    for agent, graph in aligned_graphs.items():
        is_foreign_node = partial(is_foreign_node_to_agent, agent, graph)
        remove_foreign_func = getattr(PostProcessUtils, 'remove_foreign_nodes_v%d' % post_version)
        res = remove_foreign_func(graph.copy(), is_foreign_node, problem_constants.idle_action())
        if repeat_collab:
            res = repeat_collaborative_actions(res, problem_constants.is_collaborative_action, num_repeats)
        processed_aligned_graphs[agent] = res

    for agent in processed_aligned_graphs:
        new_policy_path = os.path.join(Config.graphs_dir, problem_name)
        if exact_align:
            new_policy_path += '_exactalign'
        new_policy_path += '_postv%d' % post_version
        if repeat_collab:
            new_policy_path += '_repeatcol'
        if num_align_rounds > 1:
            new_policy_path += '_%daligns' % num_align_rounds
        if not consider_preceding_noops:
            new_policy_path += '_apn'
        if share_public_actions:
            new_policy_path += '_sharepub'
        if preprocess:
            new_policy_path += '_pre'
        new_policy_path += '_%s.dot' % agent
        result_paths.append(new_policy_path)
        networkx.nx_pydot.write_dot(processed_aligned_graphs[agent].graph,
                                    path=new_policy_path)
    return result_paths


def main(graphs=Defaults.graphs,
         suffix=Defaults.suffix,
         problem_name=Defaults.problem_name,
         problem_kind=Defaults.problem_kind,
         output_name=Defaults.output_problem):
    problem_path = os.path.join(Config.problems_dir, problem_name + Config.problem_format)

    t0 = time()

    problem = POMDPXProblemFactory.create_pomdpx_problem(problem_path, problem_kind)
    problem_constants = POMDPXProblemFactory.get_problem_constants_by_kind(problem_kind)
    num_agents = problem.num_agents

    index_to_agent = {i: problem_constants.agent_symbol(i) for i in range(num_agents)}
    agent_to_num_obsvars = {
        agent: sum([1 for obs_var in problem.obs_vars if agent in obs_var or
                    all([coagent not in obs_var for coagent in index_to_agent.values() if coagent != agent])]) for agent
        in
        index_to_agent.values()
    }
    sorted_agents = [index_to_agent[key] for key in sorted(index_to_agent.keys())]

    if graphs is None:
        graphs = [
            os.path.join(Config.graphs_dir, problem_name + "_%s" % agent + "%s" % suffix + Config.graph_format)
            for agent in sorted_agents]
    else:
        resorted_graphs = []
        for agent in sorted_agents:
            matching_graph = [graph for graph in graphs if agent in graph][0]
            resorted_graphs.append(matching_graph)
        graphs = resorted_graphs

    possible_params = {
        'index_to_agent': [index_to_agent],
        'agent_to_num_obsvars': [agent_to_num_obsvars],
        'graph_paths': [graphs],
        'exact_align': [True],
        'post_version': [3],
        'repeat_collab': [Config.repeat_collab],
        'num_repeats': [1],
        'problem_name': [output_name],
        'problem_constants': [problem_constants],
        'share_public_actions': [False],
        'num_align_rounds': [1],
        'preprocess': [True]
    }

    params_sets = [{k: v for k, v in zip(list(possible_params.keys()), pick)} for pick in
                   product(*list(possible_params.values()))]

    aligned_graphs = None
    for params_set in params_sets:
        aligned_graphs = create_aligned_solution(**params_set)

    t1 = time()
    print("Took %f" % (t1 - t0))

    return aligned_graphs


if __name__ == "__main__":
    main(*sys.argv[1:])
