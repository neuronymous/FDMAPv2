from numpy import random

from DecPOMDPSimulator.DecPOMDPModelFactories import POMDPXModelFactory
from DecPOMDPSimulator.PolicyGraph import PolicyGraph


class SimulatorFactory:
    @staticmethod
    def create_simulator(policies_paths, problem_path, problem_kind, horizon, log_level='silent'):
        policies = []
        for policy_path in policies_paths:
            policies.append(PolicyGraph(policy_path, graph_format="SARSOP"))
        model = POMDPXModelFactory().create_model(pomdpx_path=problem_path, problem_kind=problem_kind)
        return Simulator(policies=policies, model=model, horizon=horizon, log_level=log_level)


class Simulator:
    LOG_LEVELS = ['silent', 'info']

    def __init__(self, policies, model, horizon, log_level):
        self.policies = policies
        self.num_agents = len(self.policies)
        self.accumulated_reward = 0.0
        self.last_reward = 0.0
        self.horizon = horizon
        self.ticks_left = self.horizon
        self.model = model
        self.state = self.model.sample_from_initial_state()
        if log_level not in self.LOG_LEVELS:
            raise Exception("Invalid log level")
        self.log_level = log_level  # Silent, Info
        self.last_joint_action = ''
        self.last_joint_obs = ''

    def tick(self):
        if self.ticks_left == 0:
            self.terminate()
        random_tokens = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(self.num_agents)]
        joint_action = []
        for policy in self.policies:
            joint_action.append(policy.get_action())

        next_state, reward, joint_observation = self.model.step(cur_state=self.state, joint_action=joint_action,
                                                                random_tokens=random_tokens)

        self.last_joint_action = joint_action
        self.last_joint_obs = str([obs for aobs in joint_observation for obs in aobs if obs != 'null'])  # TODO use cons
        if self.log_level == self.LOG_LEVELS[1]:
            self.log_stats(joint_action, joint_observation)

        self.state = next_state
        self.last_reward = reward
        self.accumulated_reward += reward
        for i in range(self.num_agents):
            self.policies[i].update_by_obs(joint_observation[i])

        if self.ticks_left > 0:
            self.ticks_left -= 1

    def terminate(self):
        self.log_stats()

    def reset(self):
        self.state = self.model.sample_from_initial_state()
        for policy in self.policies:
            policy.reset()
        self.last_reward = 0
        self.accumulated_reward = 0
        self.ticks_left = self.horizon

    def log_stats(self, joint_action, joint_observation):
        print(tuple(self.state.values()), joint_action, joint_observation)
