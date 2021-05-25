import time
from collections import Counter, defaultdict

import networkx
import numpy as np
import os
import sys

from TeamProblemProjector.defaults import Defaults
from conf import Config

sys.path.append('..')

from GraphAlignment.BFS import BFS
from POMDPX.POMDPXConstants import ProjectionMode
from POMDPX.POMDPXFactory import POMDPXProblemFactory
from TeamProblemProjector.TraceAnalysisUtils import get_agent_contexed_actions_mapping, extract_relevant_actions_lists, \
    extract_trace_stats, calculate_precision, get_affecting_state_action_costs_and_sim_rewards
from TeamProblemProjector.SARSOPTraceUtils import parse_trace_file as trace_parser

CA_REWARD_VARIABLE = 'reward_shaping'


def project_and_sort_cas(agent, problem_pomdpx, trace_path, full_prior_state):
    is_own_public_action = lambda a: problem_pomdpx.utils.is_public_action(a) and \
                                     agent in problem_pomdpx.utils.extract_agents_from_action(a)
    agent_ca_list = extract_relevant_actions_lists(trace_path, trace_parser, is_own_public_action)
    dependency_graph = networkx.DiGraph()
    for idx, action_list in enumerate(agent_ca_list):
        prev_node_id = None
        for action, state in action_list:
            projected_context = ' '.join(
                problem_pomdpx.project_instance_v2(full_prior_state, state.split(' '), action, agent))
            curr_node_id = action + "%" + projected_context
            if curr_node_id not in dependency_graph.nodes():
                dependency_graph.add_node(curr_node_id)
                if prev_node_id is not None:
                    dependency_graph.add_edge(prev_node_id, curr_node_id)
            elif prev_node_id is not None and prev_node_id != curr_node_id:
                if not networkx.has_path(dependency_graph, curr_node_id, prev_node_id):
                    dependency_graph.add_edge(prev_node_id, curr_node_id)
                else:
                    # print("Connecting %s to %s will form a cycle" % (prev_node_id, curr_node_id))
                    pass
            prev_node_id = curr_node_id

    return [tuple(node_id.split('%')) for node_id in
            networkx.algorithms.dag.topological_sort(dependency_graph)]


def assign_rewards_to_pcas(sorted_pcas, problem_pomdpx, trace_path, success_verification,
                           tie_breaker_amplifier):
    pcas_rewards = {}
    max_cost, max_trace_len, max_public_actions_gap, total_reward = extract_trace_stats(trace_path,
                                                                                        trace_parser,
                                                                                        measured_actions=list(
                                                                                            map(lambda x: x[0],
                                                                                                sorted_pcas)),
                                                                                        is_idle_action=problem_pomdpx.utils.is_idle_action)
    print(max_cost, max_trace_len, max_public_actions_gap)

    sp = 0.8  # TODO for now while it is fixed
    min_action_succ_rate = sp  # TODO
    last_action_succ_rate = sp  # TODO projected_problem.succ_rate(sorted_contexed_actions[-1][0])
    gamma = problem_pomdpx.discount

    reward_shape_base = (total_reward / float(len(sorted_pcas))) / min_action_succ_rate

    n1 = max_trace_len - 1 - max_public_actions_gap
    n2 = max_trace_len - 1
    max_expected_preceding_cost = ((gamma ** n1 - gamma ** (n2 + 1)) / (1 - gamma)) * (
            max_cost / min_action_succ_rate)
    epsilon = 0.1

    orig_base_reward = ((max_expected_preceding_cost + epsilon) / (
            last_action_succ_rate * (gamma ** (max_trace_len - 1))))

    print(orig_base_reward)
    base_reward = max(orig_base_reward, reward_shape_base)
    print("base_reward won" if reward_shape_base != max(base_reward, reward_shape_base) else "reward_shape_base won")
    pcas_rewards[sorted_pcas[-1]] = base_reward
    prev_reward = base_reward

    for action, context in reversed(sorted_pcas[:-1]):
        curr_action_succ_rate = sp  # TODO projected_problem.succ_rate(action)
        prefer_even_if_failed_reward = (prev_reward / (
                1 - curr_action_succ_rate)) * gamma * last_action_succ_rate + epsilon if success_verification else 0
        tie_break_reward = prev_reward * tie_breaker_amplifier

        # prefer_over_reorderings_reward = prev_reward + \
        #                                  (prev_reward * last_action_succ_rate / curr_action_succ_rate) * gamma

        chosen_reward = max(prefer_even_if_failed_reward, tie_break_reward)
        pcas_rewards[(action, context)] = chosen_reward

        prev_reward = chosen_reward
        last_action_succ_rate = curr_action_succ_rate

    required_precision = calculate_precision(max_trace_len, problem_pomdpx.discount, max_public_actions_gap, max_cost,
                                             min_action_succ_rate, min_action_succ_rate,
                                             orig_base_reward)
    return pcas_rewards, required_precision


def calc_agent_ca_rewards(agent, state_action_cost_per_sim, total_cost_per_sim, total_reward_per_sim, problem_pomdpx,
                          preprojection_context):
    agent_state_action_cost_per_sim = []

    for seq in state_action_cost_per_sim:
        cur_cost = 0.0
        filtered_seq = []
        for (state, action, cost) in seq:
            cur_cost += cost
            if agent in problem_pomdpx.utils.extract_agents_from_action(action):
                filtered_seq.append((state, action, cur_cost))
                cur_cost = 0.0
        agent_state_action_cost_per_sim.append(filtered_seq)

    agent_total_cost_per_sim = [sum([x[2] for x in sim]) for sim in agent_state_action_cost_per_sim]

    agent_state_action_weight_per_sim = [
        [(s, a, (c / agent_total_cost_per_sim[sim_idx]))
         for s, a, c in agent_state_action_cost_per_sim[sim_idx]]
        for sim_idx in range(len(agent_total_cost_per_sim))]

    agent_total_reward_per_sim = list(map(lambda x: ((x[0] + 1e-10) / (x[1] + 1e-10)) * x[2],
                                          zip(agent_total_cost_per_sim, total_cost_per_sim,
                                              total_reward_per_sim)))

    # Per sim, triplets of state_action_reward of useful public actions
    agent_state_action_reward_per_sim = [[(s, a, agent_total_reward_per_sim[sim_idx] * w) for s, a, w in sim]
                                         for sim_idx, sim in enumerate(agent_state_action_weight_per_sim)]

    ca_total_rewards = Counter()
    ca_sim_appearances = Counter()
    ca_total_appearances = Counter()
    one_per_sim_appearance = False
    for sim in agent_state_action_reward_per_sim:
        sim_ca_rewards = Counter()
        cas = set()
        for state, action, reward in sim:
            context = tuple(problem_pomdpx.project_instance_v2(preprojection_context, state, action, agent))
            ca = (context, action)
            cas.add(ca)
            sim_ca_rewards[ca] += reward  # * len(problem_pomdpx.utils.extract_agents_from_action(action))
            ca_total_appearances[ca] += 1

        for ca in cas:
            ca_sim_appearances[ca] += 1
            ca_total_rewards[ca] += sim_ca_rewards[ca]

    ca_counts = ca_sim_appearances if one_per_sim_appearance else ca_total_appearances

    max_count = max(ca_counts.values())
    avg_count = np.mean(list(ca_counts.values()))
    std_count = np.std(list(ca_counts.values()))
    outlier_count = avg_count - 2 * std_count

    for ca in ca_total_rewards:
        multiplier = 0
        if ca_counts[ca] > max_count * 0.01:
            multiplier += 1 / ca_counts[ca]
        ca_total_rewards[ca] *= multiplier

    # extremely small rewards shouldn't cancel penalty.
    ca_total_rewards = {ca: reward for ca, reward in ca_total_rewards.items() if
                        reward > sum(list(ca_total_rewards.values())) * 1e-5}

    return ca_total_rewards


def is_public_var(var, problem_pomdpx):
    for action in problem_pomdpx.actions:
        if problem_pomdpx.utils.is_public_action(action) and var in problem_pomdpx.utils.extract_objectives_from_action(
                action, problem_pomdpx.metadata):
            return True
    return False


def project_public_context(variables, instance, problem_pomdpx):
    res = []
    if len(variables) != len(instance):
        raise Exception("Variables and Instance should be of same length")

    for idx, var in enumerate(variables):
        if var not in problem_pomdpx.action_vars and is_public_var(var, problem_pomdpx):
            res.append('*')
        else:
            res.append(instance[idx])
    return res


def project_team_problem(team_problem_path, team_problem_kind, trace_path, output_dir, suffix=""):
    projected_problem_path_to_precision = {}

    team_problem_pomdpx = POMDPXProblemFactory.create_pomdpx_problem(team_problem_path, team_problem_kind)

    # Get expected costs for actions in simulations that are useful (rewarding/state bringing)
    state_action_cost_per_sim, total_reward_per_sim = get_affecting_state_action_costs_and_sim_rewards(
        trace_path=trace_path,
        trace_parser=trace_parser,
        problem_pomdpx=team_problem_pomdpx)  # TODO OPTIMIZE

    total_cost_per_sim = [sum([x[2] for x in sim]) for sim in state_action_cost_per_sim]

    all_applied_actions = set([a for sim in state_action_cost_per_sim for s, a, r in sim])
    unused_public_action = set([a for a in team_problem_pomdpx.actions if
                                team_problem_pomdpx.utils.is_public_action(a) and a not in all_applied_actions])

    agents = team_problem_pomdpx.agents
    epsilon = 1

    # state at time zero
    full_time_zero_state = [*list(team_problem_pomdpx.action_vars.keys())] + \
                           [state_var.prev for state_var in team_problem_pomdpx.state_vars.values()]

    # state at time one
    full_time_one_state = [*list(team_problem_pomdpx.action_vars.keys())] + \
                          [state_var.curr for state_var in team_problem_pomdpx.state_vars.values()]

    full_state = [*list(team_problem_pomdpx.action_vars.keys())] + \
                 [state_var.name for state_var in team_problem_pomdpx.state_vars.values()]

    agent_to_ca_rewards = {
        agent: calc_agent_ca_rewards(agent=agent, state_action_cost_per_sim=state_action_cost_per_sim,
                                     total_cost_per_sim=total_cost_per_sim,
                                     total_reward_per_sim=total_reward_per_sim,
                                     problem_pomdpx=team_problem_pomdpx,
                                     preprojection_context=full_time_zero_state) for agent in agents}

    for agent in agents:

        # init with team problem
        projected_problem = POMDPXProblemFactory.create_pomdpx_problem(team_problem_path, team_problem_kind)

        agent_public_objectives = set()
        agent_private_objectives = set()

        agent_applied_actions = set(
            filter(lambda action: agent in projected_problem.utils.extract_agents_from_action(action),
                   all_applied_actions))
        # Extract all agent objectives that derive from applied public actions
        for action in agent_applied_actions:
            action_objectives = team_problem_pomdpx.utils.extract_objectives_from_action(action=action, metadata=team_problem_pomdpx.metadata)
            agent_public_objectives = agent_public_objectives.union(action_objectives)

        # Extract all agent objectives that appears in private actions (non-sense and non-public)
        for action in projected_problem.actions:
            if (not team_problem_pomdpx.utils.is_public_action(action) and
                not team_problem_pomdpx.utils.is_sense_action(action)) and (
                    agent in team_problem_pomdpx.utils.extract_agents_from_action(action)):
                for objective in team_problem_pomdpx.utils.extract_objectives_from_action(action,
                                                                                          team_problem_pomdpx.metadata):
                    agent_private_objectives.add(objective)

        # Remove reward variable with own public-objective, or foreign objective (private of co-agent)
        rewards_funcs_to_delete = []
        for k, func in projected_problem.rewards_func.items():
            func_objective = func.subject
            # Ignore non-objective reward functions
            if func_objective not in agent_public_objectives:
                continue
            rewards_funcs_to_delete.append(k)
        for k in rewards_funcs_to_delete:
            projected_problem.rewards_func.pop(k)
            projected_problem.reward_vars.pop(k)

        # Delete unused public actions
        for action in unused_public_action:
            projected_problem.delete_action(action)

        # Delete observation variables of other agents
        obs_vars_to_delete = [obs_var for obs_var in projected_problem.observations_func if
                              (len(projected_problem.utils.extract_agents_from_obsvar(obs_var)) > 0) and
                              agent not in projected_problem.utils.extract_agents_from_obsvar(obs_var)]
        for obs_var in obs_vars_to_delete:
            projected_problem.delete_obs_var(obs_var)

        # Remove other agents sensing actions
        for action in projected_problem.actions.copy():
            if projected_problem.utils.is_sense_action(action) and \
                    agent not in projected_problem.utils.extract_agents_from_action(action):
                projected_problem.delete_action(action)

        # Turn all co-agents private action to deterministic
        for action in projected_problem.actions.copy():
            if not projected_problem.utils.is_public_action(action) and \
                    agent not in projected_problem.utils.extract_agents_from_action(action):
                projected_problem.determinize_action(action)

        # Create new reward func for contexted rewarding and penalizing
        projected_problem.add_reward_var(CA_REWARD_VARIABLE)
        projected_problem.add_reward_func(var=CA_REWARD_VARIABLE,
                                          parents_list=full_time_zero_state)
        ca_reward_func = projected_problem.rewards_func[CA_REWARD_VARIABLE]

        ca_rewards = agent_to_ca_rewards[agent]

        precision = Config.sa_precision
        hard_penalty = sum(ca_rewards.values()) + epsilon

        # By default penalize public actions with the hard penalty
        for pa in all_applied_actions:
            instance_string = pa + " *" * (len(full_time_zero_state) - 1)
            projected_problem.add_entry_to_func(func=ca_reward_func,
                                                instance_string=instance_string,
                                                table=np.array(-hard_penalty))

        # Cancel penalty for public context
        for ca, reward in ca_rewards.items():
            context, action = ca
            public_context = project_public_context(full_state, context, projected_problem)
            projected_problem.add_entry_to_func(func=ca_reward_func, instance_string=' '.join(public_context),
                                                table=np.array(0.0))

        # Gap fillers, cancel penalty for public context
        for co_agent in agents:
            used_contexts = []
            if co_agent != agent:
                co_agent_ca_rewards = agent_to_ca_rewards[co_agent]
                for ca in co_agent_ca_rewards:
                    context, action = ca
                    if context in used_contexts:
                        continue
                    public_context = project_public_context(full_state, context, projected_problem)
                    projected_problem.add_entry_to_func(func=ca_reward_func, instance_string=' '.join(public_context),
                                                        table=np.array(0.0))
                    used_contexts.append(context)

        # Add rewards for exact context
        for ca, reward in ca_rewards.items():
            context, action = ca
            projected_problem.add_entry_to_func(func=ca_reward_func, instance_string=' '.join(context),
                                                table=np.array(reward))

        new_problem_path = os.path.join(output_dir, (team_problem_path[:-len('.pomdpx')] + "_" + agent).split('/')[-1])

        if suffix is not None and suffix != '':
            new_problem_path += '_%s' % suffix

        new_problem_path += '.pomdpx'
        projected_problem.write_problem(new_problem_path)
        projected_problem_path_to_precision[new_problem_path] = precision
    print(projected_problem_path_to_precision)
    return projected_problem_path_to_precision


def main(problem_name=Defaults.problem_name, problem_kind=Defaults.problem_kind, suffix=Defaults.suffix):
    trace_path = os.path.join(Config.traces_dir, problem_name + Config.trace_format)
    problem_path = os.path.join(Config.problems_dir, problem_name + Config.problem_format)
    output_dir = Config.problems_dir

    t0 = time.time()
    projected_problem_path_to_precision = project_team_problem(team_problem_path=problem_path,
                                                               team_problem_kind=problem_kind,
                                                               trace_path=trace_path,
                                                               output_dir=output_dir,
                                                               suffix='')

    t1 = time.time()
    print("Projection took %f" % (t1 - t0))
    return projected_problem_path_to_precision


if __name__ == "__main__":
    main(*sys.argv[1:])
