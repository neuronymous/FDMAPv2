from collections import defaultdict

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
                    cur_state = tuple(content.split(","))
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