from collections import Counter

import numpy as np
import os
from lxml import etree as ET
import re

from BP.POMDPX import POMDPXProblem
from BP.TraceAnalyzing import get_agent_to_sa_mapping

PROBLEMS_DIR = '../Resources/problems'
TRACES_DIR = '../Resources/traces'
PROBLEM_NAME = 'BP-3x3_2A_2H_1L_HARD_MINOBS'
PROBLEM_PATH = os.path.join(PROBLEMS_DIR, PROBLEM_NAME) + '.pomdpx'
TRACE_PATH = os.path.join(TRACES_DIR, PROBLEM_NAME) + '.trace'
REWARDING_PUBLIC_ACTIONS_PREFIXES = ['ap', 'ajp']
PENALIZED_PUBLIC_ACTIONS_PREFIXES = ['as']

PUBLIC_ACTIONS_REWARD_VNAME = "public_actions_reward"
CONTEXTED_PUBLIC_ACTION_REWARD = 200


def create_element(*args, text=None, **kwargs):
    res = ET.Element(*args, **kwargs)
    if text:
        res.text = text
    res.tail = '\n'
    return res


def is_action_public(action):
    return any([bool(re.match(prefix + ".*", action)) for prefix in REWARDING_PUBLIC_ACTIONS_PREFIXES])


def calc_projected_instance(problem, parents, state, action):
    """BoxPushing SPECIFIC!"""
    res = []
    state_elements = state.split(' ')
    state_idx = 0
    action_subjects = [elem for elem in action.split('_') if elem.startswith('b') and elem in problem.state_vars]
    action_agents = [elem for elem in action.split('_') if elem.startswith('a') and elem in problem.agents]

    for parent in parents:
        if parent in problem.action_vars:
            res.append(action)
        else:  # parent is a state var
            should_project = True
            for agent in action_agents:
                if agent in parent:
                    should_project = False
                    break
            for subject in action_subjects:
                if subject in parent:
                    should_project = False
                    break

            if should_project:
                res.append('*')
            else:
                res.append(state_elements[state_idx])
            state_idx += 1

    return ' '.join(res)


def project_team_problem_by_mapping(team_problem_path, rewarded_agent_action_state_mapping,
                                    reward, penalized_agent_action_state_mapping=None,
                                    penalty_multiplier=1.1, delete_unused_public=False, add_gap_filler_actions=False,
                                    delete_original_objectives=False, delete_coagents_sense=False,
                                    project_public_actions=False,
                                    special_suffix=None, determinize_gap_fillers=False):
    """
    agent action state mapping maps agent to set of actions, each action mapped to a set of states
    reward is the base reward given for public actions
    penalty_multiplier determines the default penalty for public actions,
    which will be overridden on relevant public actions
    :param delete_coagents_sense:
    :param penalized_agent_action_state_mapping:
    :param team_problem_path:
    :param rewarded_agent_action_state_mapping:
    :param reward: the reward given for public action execution
    :param penalty_multiplier: by default non contexted public actions will be penalized by multiplier*reward.
    if delete_unused_public is set to True, this param is obsolete
    :param delete_unused_public: delete all contexted public actions that were unused by the current agent,
    :param add_gap_filler_actions: add all contexted public actions that were applied by other agents for free,
    :param delete_original_objectives: deletes all reward funcs that their subject is in the problem's objectives
    :return:
    """
    """"""

    for agent, action_state_map in rewarded_agent_action_state_mapping.items():
        problem = POMDPXProblem()
        problem.parse_problem(team_problem_path)
        full_prior_state_string = ' '.join(
            [action_var for action_var in problem.action_vars] + [state_var.prev for state_var in
                                                                  problem.state_vars.values()])

        relevant_objectives = set()
        objectives_relevancy = Counter()

        # Count the number of states from which each action is performed.
        # We want peripheral objectives to become irrelevant enough so they won't be acted on in an action chain.
        # Might prefer to remove objective rewards alltogether
        # TODO think about isolated yet necessary actions
        for action in action_state_map:
            objectives = set(action.split('_')).intersection(set(problem.objectives))
            for objective in objectives:
                objectives_relevancy[objective] += len(action_state_map[action])
            relevant_objectives = relevant_objectives.union(objectives)

        # Each objective is associated with the relative number of times it was referenced in public actions
        normalizing_factor = sum(objectives_relevancy.values())
        for objective in objectives_relevancy:
            objectives_relevancy[objective] /= normalizing_factor

        if delete_original_objectives:
            rewards_funcs_to_delete = []
            for k, func in problem.rewards_func.items():
                func_objective = func.subject
                if func_objective not in problem.objectives:  # Ignore non-objective reward functions
                    continue
                rewards_funcs_to_delete.append(k)
            for k in rewards_funcs_to_delete:
                problem.rewards_func.pop(k)
                problem.reward_vars.pop(k.strip())
        else:
            # Remove irrelevant objectives, and change the rewards for existing objectives based on their relevancy
            for func in problem.rewards_func.values():
                func_objective = func.subject
                # Ignore non-objective reward functions (e.g. actions costs)
                if func_objective not in problem.objectives:
                    continue

                multiplier = 0  # Default
                if func_objective in relevant_objectives:
                    multiplier = objectives_relevancy[func_objective]

                entries = func.entries
                for entry in entries:
                    if np.min(entry.table) >= 0:
                        entry.table *= multiplier

        # Create new reward func for contexted rewarding and penalizing
        public_actions_reward_vname = PUBLIC_ACTIONS_REWARD_VNAME
        problem.add_reward_var(public_actions_reward_vname)

        public_actions = [action for action in problem.actions if
                          any([action.startswith(prefix) for prefix in
                               REWARDING_PUBLIC_ACTIONS_PREFIXES])]

        problem.add_reward_func(var=public_actions_reward_vname, parents_list=full_prior_state_string.split(' '))
        reward_func = problem.rewards_func[public_actions_reward_vname]

        # By default penalize public actions, to avoid positive reward abuse
        penalty = np.array(-penalty_multiplier * reward)
        for pa in public_actions:
            instance_string = pa + " *" * (len(full_prior_state_string.split(' ')) - 1)
            problem.add_entry_to_func(func=reward_func, instance_string=instance_string, table=penalty)

        if penalized_agent_action_state_mapping is not None:
            uncertainty_constant = 5
            penalty = np.array((-penalty_multiplier * reward) / uncertainty_constant)
            for pa in penalized_agent_action_state_mapping[agent]:
                instance_string = pa + " *" * (len(full_prior_state_string.split(' ')) - 1)
                problem.add_entry_to_func(func=reward_func, instance_string=instance_string, table=penalty)

        # Add reward for public action in context
        contexted_public_action_reward = np.array(reward)
        for public_action in action_state_map:
            used_projections = []
            for state in action_state_map[public_action]:
                if not project_public_actions:
                    instance_string = public_action + ' ' + state
                else:
                    instance_string = calc_projected_instance(problem, reward_func.parents, state, public_action)
                    if instance_string in used_projections:
                        continue
                    used_projections.append(instance_string)
                problem.add_entry_to_func(func=reward_func, instance_string=instance_string,
                                          table=contexted_public_action_reward)

        # Cancel the penalty for contexted application
        if penalized_agent_action_state_mapping is not None:
            penalized_public_action_reward = np.array(0)
            penalized_action_state_map = penalized_agent_action_state_mapping[agent]
            for public_action in penalized_action_state_map:
                for state in penalized_action_state_map[public_action]:
                    instance_string = public_action + ' ' + state
                    problem.add_entry_to_func(func=reward_func, instance_string=instance_string,
                                              table=penalized_public_action_reward)

        gap_fillers = {}
        if add_gap_filler_actions:
            gap_fillers = {a: s_set for co_agent in rewarded_agent_action_state_mapping.keys() if co_agent != agent
                           for a, s_set in rewarded_agent_action_state_mapping[co_agent].items()}

            for a, s_set in gap_fillers.items():
                used_projections = []
                for state in s_set:
                    # We only add the gap filler if it wasn't already rewarded (joint actions appear in both)
                    if a not in action_state_map or state not in action_state_map[a]:
                        if not project_public_actions:
                            instance_string = a + ' ' + state
                        else:
                            instance_string = calc_projected_instance(problem, reward_func.parents, state,
                                                                      a)
                            if instance_string in used_projections:
                                continue
                            used_projections.append(instance_string)

                        gap_filler_reward = np.array(0)
                        problem.add_entry_to_func(func=reward_func, instance_string=instance_string,
                                                  table=gap_filler_reward)
                if determinize_gap_fillers:
                    problem.determinize_action(a)

        if delete_unused_public:
            for action in problem.actions.copy():
                if action not in action_state_map.keys() and action not in gap_fillers and is_action_public(action):
                    problem.delete_action(action)

        if delete_coagents_sense:
            # in minobs agent is not part of the obsvar
            obs_vars_to_delete = [obs_var for obs_var in problem.observations_func if
                                  (bool(re.match('.*_a\d$', obs_var))) and agent not in obs_var]
            for obs_var in obs_vars_to_delete:
                problem.delete_obs_var(obs_var)

            # remove other agents sensing actions
            for action in problem.actions.copy():
                if problem.is_sense_action(action) and agent not in action:  # TODO move regex stuff to POMDPX
                    problem.delete_action(action)

        new_problem_path = team_problem_path[:-len('.pomdpx')] + "_" + agent
        if penalized_agent_action_state_mapping is not None:
            new_problem_path += '_PENALIZESENSE'
        if determinize_gap_fillers:
            new_problem_path += '_DETGF'
        new_problem_path += '_R%d' % reward
        if special_suffix is not None:
            new_problem_path += '_%s' % special_suffix

        new_problem_path += ".pomdpx"
        problem.write_problem(new_problem_path)


rewarded_agent_to_sa_map = get_agent_to_sa_mapping(TRACE_PATH, REWARDING_PUBLIC_ACTIONS_PREFIXES)
project_team_problem_by_mapping(PROBLEM_PATH, rewarded_agent_to_sa_map, penalty_multiplier=500,
                                reward=CONTEXTED_PUBLIC_ACTION_REWARD,
                                add_gap_filler_actions=True,
                                delete_unused_public=True,
                                delete_original_objectives=True,
                                delete_coagents_sense=True,
                                project_public_actions=True,
                                )
