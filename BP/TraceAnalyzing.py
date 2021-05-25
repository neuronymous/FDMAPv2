import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import os

# TRACE_PATHS = ["./BP-3x2_3A_1H_2L_LOWCOST.trace",
#                "./BP-3x2_3A_1H_2L_HIGHREWARD.trace",
#                "./BP-3x2_3A_1H_2L.trace",
#                "./BP-3x2_3A_1H_2L_RELAXEDCOST.trace"]
TRACE_FILES = ["./BP-3x2_3A_0H_3L_EXPLICITOBJ_a1_DELUNUSED_GAPFILLERS_DELGOALS.trace",
               "./BP-3x2_3A_0H_3L_EXPLICITOBJ_a1_DELUNUSED_GAPFILLERS_DELGOALS_DELSENSE.trace"]
TRACES_DIR = "../Resources/traces"
TRACE_PATHS = [os.path.join(TRACES_DIR, TRACE_FILE) for TRACE_FILE in TRACE_FILES]

SIMULATION_END_TOKEN = "----- time"
SIMULATION_START_TOKEN = ">>> begin"
REWARD_TOKEN = "R"
ACTION_TOKEN = "A"
STATE_TOKEN = "Y"

ACTIONS_TO_COUNT = ['ap', 'ajp', 'as']


class Simulation:
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


class Step:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = float(reward)


def parse_trace_file(trace_path):
    simulations = []

    cur_simulation = Simulation()
    cur_action, cur_state, cur_reward = None, None, None

    with open(trace_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            try:
                if line.startswith(SIMULATION_END_TOKEN):
                    simulations.append(cur_simulation)
                    cur_simulation = Simulation()

                line = line.replace(" ", "")
                cur_token, content = line.split(':')
                content = content.strip('()\r\n')

                if cur_token == ACTION_TOKEN:
                    cur_action = content
                elif cur_token == STATE_TOKEN:
                    cur_state = content
                elif cur_token == REWARD_TOKEN:
                    cur_reward = content

                if cur_state and cur_action and cur_reward:
                    cur_step = Step(cur_state, cur_action, cur_reward)
                    cur_simulation.add_step(cur_step)
                    cur_action, cur_state, cur_reward = None, None, None
            except ValueError:
                pass

    return simulations


def calculate_total_action_count(simulations):
    res = defaultdict(int)
    for sim in simulations:
        for action, count in sim.action_count.items():
            res[action] += count

    return sorted(res.items(), key=operator.itemgetter(1), reverse=True)


def calculate_avg_action_rewards(simulations):
    res = defaultdict(float)
    for sim in simulations:
        cur_action_rewards = sim.get_action_rewards()
        for action, reward in cur_action_rewards.items():
            res[action] += reward

    action_counts = dict(calculate_total_action_count(simulations))

    for key, val in res.items():
        res[key] = val / action_counts[key]

    res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
    return res


def get_filtered_total_action_count(simulations):
    return filter_actions_from_mapping(calculate_total_action_count(simulations))


def get_filtered_avg_actions_reward(simulations):
    return filter_actions_from_mapping(calculate_avg_action_rewards(simulations))


def filter_actions_from_mapping(action_mapping):
    return list(filter(lambda x: any([x[0].startswith(prefix) for prefix in ACTIONS_TO_COUNT]),
                       action_mapping))


def print_simulations_data(simulations):
    for idx, sim in enumerate(simulations):
        print(sim.get_stats())

    print(get_filtered_total_action_count(simulations))
    print(get_filtered_avg_actions_reward(simulations))


def plot_histogram(list_of_tuples, x_label=None, y_label=None, title=None):
    plt.bar(*zip(*list_of_tuples))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()


def main():
    for trace_path in TRACE_PATHS:
        simulations = parse_trace_file(trace_path)
        action_to_reward = get_filtered_avg_actions_reward(simulations)
        action_to_applications = get_filtered_total_action_count(simulations)

        print_simulations_data(simulations)

        plot_histogram(list_of_tuples=action_to_reward, x_label='Action', y_label='Avg Reward',
                       title=trace_path)

        plt.show()

        plot_histogram(list_of_tuples=action_to_applications, x_label='Action', y_label='Applications')
        plt.show()


class AgentToActionToState(dict):
    def __getitem__(self, item):
        if item not in self.keys():
            self.__setitem__(item, defaultdict(set))
        return super().__getitem__(item)


def get_agent_to_sa_mapping(bp_trace_path, public_actions_prefixes, ignore_reward_or_change=False):
    """Generate a mapping of agents to public actions to the states they were applied in
    Only rewarding or state-changing actions are considered"""

    def is_action_public(action):
        return any([bool(re.match(prefix + ".*", action)) for prefix in public_actions_prefixes])

    simulations = parse_trace_file(bp_trace_path)
    res = AgentToActionToState()
    for simulation in simulations:

        num_steps = len(simulation.steps)
        for i in range(num_steps):
            step = simulation.steps[i]
            cur_state = step.state
            next_state = simulation.steps[i + 1].state if i + 1 < len(simulation.steps) else cur_state
            cur_action = step.action

            # Add action and state if the action is public and it's application is rewarding or state-changing
            if is_action_public(cur_action) and (
                    ignore_reward_or_change or (step.reward > 0 or cur_state != next_state)):
                acting_agents = list(
                    component for component in cur_action.split('_')[1:] if
                    component.startswith('a'))  # should be taken from POMDPX object
                for agent in acting_agents:
                    res[agent][cur_action].add(step.state.replace(',', ' '))  # should be uniformed
    return res


def get_agent_to_box_mapping(bp_trace_path):
    """Generates a mapping from agents to boxes (can be generalized to any problem given actions,
     with agents and objectives in their names).
    It uses the fact that the action to reward list is ordered descendingly by the reward"""
    simulations = parse_trace_file(bp_trace_path)
    action_to_reward = get_filtered_avg_actions_reward(simulations)

    boxes = set()

    for action, _ in action_to_reward:
        actions_components = action.split('_')[1:]  # Remove the action itself
        for comp in actions_components:
            if comp.startswith('b'):
                boxes.add(comp)

    res = defaultdict(list)

    boxes_left = boxes
    while len(boxes_left) > 0 and len(action_to_reward) > 0:
        cur_action, _ = action_to_reward[0]
        box = list(filter(lambda comp: comp.startswith('b'), cur_action.split('_')[1:]))[0]  # Remove the action itself
        action_agents = set(
            filter(lambda comp: comp.startswith('a'), cur_action.split('_')[1:]))  # Remove the action itself

        if len(action_agents) == 1:
            action_agents = action_agents.pop()
        else:
            action_agents = frozenset(action_agents)

        if box in boxes_left:
            res[action_agents].append(box)
            boxes_left.remove(box)
        action_to_reward.pop(0)

    print(res)
    return res


if __name__ == "__main__":
    # get_agent_to_box_mapping(TRACE_PATHS[0])
    main()
