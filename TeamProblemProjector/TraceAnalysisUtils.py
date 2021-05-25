from typing import Any, Callable, Union

import numpy as np
from collections import defaultdict
from conf import Config
import math


class AgentToActionToState(dict):
    def __getitem__(self, item):
        if item not in self.keys():
            self.__setitem__(item, defaultdict(set))
        return super().__getitem__(item)


class Simulation():
    def __init__(self):
        self.steps = []
        self.action_count = defaultdict(int)
        self.total_reward = 0

    def add_step(self, step):
        self.steps.append(step)
        self.action_count[step.action] += 1
        self.total_reward += step.reward

    def get_stats(self):
        return {"Number of steps": len(self.steps),
                "Per Simulation Actions count": sorted(self.action_count.items(), key=operator.itemgetter(1),
                                                       reverse=True),
                "Total_reward": self.total_reward}

    def get_action_rewards(self):
        res = defaultdict(float)
        for step in self.steps:
            res[step.action] += step.reward
        return res


class TracesNumGenerator:
    def __init__(self, type_sampling: str):
        if type_sampling == "heuristic":
            self._generate = lambda: Config.num_traces
        elif type_sampling == "iterative":
            self._generate = generate_iterative_num_traces
        elif type_sampling == "cb":
            self._generate = generate_cb_num_traces

    def generate(self, *args, **kwargs):
        return self._generate(*args, **kwargs)


def get_agent_contexed_actions_mapping(trace_path, trace_parser, problem_constants):
    """Generate a mapping of agents to public actions to the states they were applied in, and the reward they yielded.
    Only rewarding or exploring (new-state) actions are considered"""

    simulations = trace_parser(trace_path)
    res = defaultdict(lambda: defaultdict(set))
    for simulation in simulations:
        visited_states = set()
        for i, step in enumerate(simulation.steps):
            cur_state = step.state
            visited_states.add(cur_state)
            next_state = simulation.steps[i + 1].state if i + 1 < len(simulation.steps) else cur_state
            cur_action = step.action
            cur_reward = step.reward

            # Add action and state if the action is public and it's application is rewarding or state-changing
            if problem_constants.is_public_action(cur_action) and (
                    cur_reward > 0 or (next_state not in visited_states)):
                actors = problem_constants.extract_agents_from_action(cur_action)
                for agent in actors:
                    res[agent][cur_action].add((cur_state, step.reward))  # TODO
    return res


def generate_cb_num_traces(k=8, alpha=0.1, beta=0.1):
    """TODO - in practice, might be overkill, better work with heuristics or move to iterative
    TODO2 - check for effective sample size"""
    condition_holds: Callable[[float], bool] = lambda m: m * beta - math.log(n) * (1 - k) >= \
                                                         (k - 1) * (1 + math.log(beta * k - 1)) - math.log(1 - alpha)
    n = int(math.ceil(k - 1 / beta))
    while not condition_holds(n):
        n += 1
    return n


def generate_iterative_num_traces():
    # TODO, sample until all initial states exist
    pass


def get_affecting_state_action_costs_and_sim_rewards(trace_path, trace_parser, problem_pomdpx):
    simulations = trace_parser(trace_path)
    sim_costs, sim_rewards = [], []
    discount = problem_pomdpx.discount
    for simulation in simulations:
        visited_states = set(problem_pomdpx.all_initial_states)
        state_action_cost_triplets = []  # state, action, cost
        curr_expected_cost = 0.0
        total_sim_reward = 0.0
        for i, step in enumerate(simulation.steps):
            cur_state_values = step.state
            visited_states.add(cur_state_values)
            next_state_values = simulation.steps[i + 1].state if i + 1 < len(simulation.steps) else cur_state_values
            curr_action = step.action
            curr_reward = step.reward

            curr_state = {k: v for k, v in zip(problem_pomdpx.state_vars.keys(), cur_state_values)}
            next_state = {k: v for k, v in zip(problem_pomdpx.state_vars.keys(), next_state_values)}

            sum_curr_rewards, all_rewards = problem_pomdpx.calc_rewards(curr_state, curr_action, next_state)
            curr_prob = problem_pomdpx.calc_transition_prob(curr_state, curr_action, next_state)

            curr_cost = -1 * sum(filter(lambda r: r < 0, all_rewards))
            curr_expected_cost += (curr_cost / ((discount ** i) * curr_prob))

            curr_reward = sum(filter(lambda r: r >= 0, all_rewards))
            curr_expected_reward = (curr_reward / ((1) * curr_prob))
            total_sim_reward += curr_expected_reward

            # Add action and state if the action is public and it's application is rewarding or creating a new state
            if problem_pomdpx.utils.is_public_action(curr_action) and (
                    curr_reward > 0 or (next_state_values not in visited_states)):
                state_action_cost_triplets.append((cur_state_values, curr_action, curr_expected_cost))
                curr_expected_cost = 0.0

        sim_costs.append(state_action_cost_triplets)
        sim_rewards.append(total_sim_reward)

    return sim_costs, sim_rewards


def extract_relevant_actions_lists(trace_path, trace_parser, is_relevant_action, ignore_reward_or_change=False):
    simulations = trace_parser(trace_path)
    res = []
    for simulation in simulations:
        cur_action_list = []
        num_steps = len(simulation.steps)
        for i in range(num_steps):
            step = simulation.steps[i]
            cur_state = step.state
            next_state = simulation.steps[i + 1].state if i + 1 < len(simulation.steps) else cur_state
            cur_action = step.action
            if is_relevant_action(cur_action) and (
                    ignore_reward_or_change or (step.reward > 0 or cur_state != next_state)):
                cur_action_list.append((cur_action, cur_state.replace(',', ' ')))
                res.append(cur_action_list)

    return res


def extract_trace_stats(trace_path, trace_parser, measured_actions, is_idle_action):
    simulations = trace_parser(trace_path)
    max_cost, max_trace_len, max_actions_gap = -1, -1, -1
    total_reward = 0

    for sim in simulations:
        curr_gap = 0
        num_consecutive_idles = 0
        curr_trace_len = 0
        for step in sim.steps:
            if step.reward < 0:
                max_cost = max(max_cost, step.reward * -1)
            else:
                total_reward += step.reward
            if is_idle_action(step.action):
                num_consecutive_idles += 1
            else:
                num_consecutive_idles = 0
            curr_trace_len += 1

            if step.action in measured_actions:
                max_actions_gap = max(max_actions_gap, curr_gap)
                curr_gap = 0
            else:
                curr_gap += 1

        curr_trace_len = curr_trace_len - num_consecutive_idles
        max_trace_len = max(max_trace_len, curr_trace_len)
    total_reward /= len(simulations)

    return max_cost, max_trace_len, max_actions_gap, total_reward


def calculate_precision(max_trace_length, discount, max_gap_length, max_cost, base_success_prob,
                        min_action_success_prob,
                        base_reward):
    return (discount ** max_trace_length) * base_success_prob * base_reward - sum(
        [(max_cost) * (discount ** (max_trace_length - max_gap_length + i)) for i in
         range(max_gap_length + 1)])


def calculate_precision2(max_trace_length, discount, max_gap_length, max_cost, base_success_prob,
                         min_action_success_prob,
                         base_reward):
    return (discount ** (max_trace_length - 1)) * base_reward - sum(
        [(max_cost / min_action_success_prob) * (discount ** (max_trace_length - max_gap_length + i)) for i in
         range(max_gap_length)])


def calculate_precision3(max_trace_length, discount, max_gap_length, max_cost, base_success_prob,
                         min_action_success_prob,
                         base_reward, rewards):
    return ((discount ** (max_trace_length - 1)) * base_success_prob * (
            sum([r for i, r in enumerate(rewards)]) / len(rewards))) \
           - sum([((max_cost / min_action_success_prob) * (discount ** i)) for i in
                  range(max_gap_length)])
