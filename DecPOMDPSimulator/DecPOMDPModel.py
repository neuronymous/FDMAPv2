import itertools
from abc import ABC, abstractmethod

# state is a dict that maps state var to value
# initial belief is a set of pairs of (distribution, partial_states)
from builtins import zip
from collections import OrderedDict
from functools import reduce, partial

from numpy import random
import numpy as np

from DecPOMDPSimulator.POMDPXMarks import POMDPXMarks
from DecPOMDPSimulator.Distribution import Distribution


class RealNumbers:
    def __contains__(self, item):
        return isinstance(item, float)


class Var:
    def __init__(self, name, values):
        self.name = name
        self.domain = values  # Order is important

    def is_valid_value(self, value):
        return value in self.domain


class Func:
    parent = None  # tuple of vars
    entries = None  # list of entries
    var = None  # var it affects

    class Entry:
        instance = None  # tuple of objects (var values or special characters)
        table = None  # numpy matrix

        def __init__(self, instance, table):
            self.instance = instance
            self.table = table

    def __init__(self, var, parent):
        self.WILDCARD_SIGN = POMDPXMarks.WILDCARD_SIGN
        self.ITERATION_SIGN = POMDPXMarks.ITERATION_SIGN
        self.var = var
        self.parent = parent
        self.entries = []

    def get_value(self, parent_arguments):
        parent_len = len(self.parent)
        if len(parent_arguments) != parent_len:
            raise Exception("Invalid number of arguments")
        for i in range(len(self.entries) - 1, -1, -1):
            cur_entry = self.entries[i]
            table_indices = []
            entry_matches = True
            for idx in range(parent_len):
                entry_instance_value = cur_entry.instance[idx]
                argument = parent_arguments[idx]
                if self.is_special_character(entry_instance_value):
                    if self.is_wildcard(entry_instance_value):
                        continue
                    elif self.is_iteration(entry_instance_value):
                        table_indices.append(self.parent[idx].domain.index(argument))
                elif entry_instance_value != argument:
                    entry_matches = False
                    break
            if entry_matches:
                # m = len(table_indices)
                # n = len(cur_entry.table.shape)
                # for _ in range(n - m):
                #     table_indices.append(0)
                return np.squeeze(cur_entry.table)[tuple(table_indices)]

        return 0.0

    def create_instance(self, phase_overseer, prev_vars=None, next_vars=None, current_vars=None):
        res = []
        all_vars = self.parent.copy()
        all_vars.append(self.var)
        for var in all_vars:
            var_name = var.name
            var_phase = phase_overseer.get_var_phase(var_name)
            cur_var = phase_overseer.get_unphased_var(var_name)
            if var_phase == phase_overseer.prev and (prev_vars is not None and cur_var in prev_vars):
                res.append(prev_vars[cur_var])
            elif var_phase == phase_overseer.current and (current_vars is not None and cur_var in current_vars):
                res.append(current_vars[cur_var])
            elif var_phase == phase_overseer.next and (next_vars is not None and cur_var in next_vars):
                res.append(next_vars[cur_var])
            elif var_name != self.var.name:  # In case we're in the last var, we allow for no match
                raise Exception("No phase detected for var, shouldn't get here")

        return res

    def is_special_character(self, entry_instance_value):
        return self.is_iteration(entry_instance_value) or self.is_wildcard(entry_instance_value)

    def is_iteration(self, entry_instance_value):
        return entry_instance_value == self.ITERATION_SIGN

    def is_wildcard(self, entry_instance_value):
        return entry_instance_value == self.WILDCARD_SIGN

    def create_and_add_entry(self, instance, table):
        self.entries.append(Func.Entry(instance=instance, table=table))


class ProbFunc(Func):
    def _get_dist(self, parent_arguments):
        parent_len = len(self.parent)
        total_prob = 0.0
        dist_creation_dict = {}
        for i in range(len(self.entries) - 1, -1, -1):
            cur_entry = self.entries[i]
            table_indices = []
            entry_matches = True
            for idx in range(parent_len):
                entry_instance_value = cur_entry.instance[idx]
                argument = parent_arguments[idx]
                if self.is_special_character(entry_instance_value):
                    if self.is_wildcard(entry_instance_value):
                        continue
                    elif self.is_iteration(entry_instance_value):
                        table_indices.append(self.parent[idx].domain.index(argument))
                elif entry_instance_value != argument:
                    entry_matches = False
                    break

            if entry_matches:
                var_value = cur_entry.instance[-1]
                if self.is_iteration(var_value):  # distribution of var's domain
                    # m = len(table_indices)
                    # n = len(cur_entry.table.shape)
                    # for _ in range(n - m - 1):
                    #     table_indices.append(0)
                    table_indices.append(slice(None))
                    probabilities = np.squeeze(cur_entry.table)[tuple(table_indices)]
                    total_prob += sum(probabilities)
                    # probabilities matches the func's var values only, no >1 dimension distributions
                    dist_creation_dict = {**dist_creation_dict,
                                          **{v: p for v, p in zip(self.var.domain, probabilities)}}
                else:  # exact prob specification
                    # m = len(table_indices)
                    # n = len(cur_entry.table.shape)
                    # for _ in range(n - m):
                    #     table_indices.append(0)
                    prob = np.squeeze(cur_entry.table)[tuple(table_indices)]
                    total_prob += prob
                    dist_creation_dict[var_value] = prob

                if abs(total_prob - 1.0) < Distribution.epsilon:  # Perhaps we can't really not be in this case
                    return Distribution(dist_creation_dict)
        return None

    def _get_exact_prob(self, full_arguments):
        full_instance_len = len(self.parent) + 1
        for i in range(len(self.entries) - 1, -1, -1):
            cur_entry = self.entries[i]
            table_indices = []
            entry_matches = True
            for idx in range(full_instance_len):
                entry_instance_value = cur_entry.instance[idx]
                argument = full_arguments[idx]
                is_last_instance_value = (idx == full_instance_len - 1)
                if self.is_special_character(entry_instance_value):
                    if self.is_wildcard(entry_instance_value):
                        continue
                    elif self.is_iteration(entry_instance_value):
                        table_idx = self.var.domain.index(argument) if is_last_instance_value else \
                            self.parent[idx].domain.index(argument)
                        table_indices.append(table_idx)
                elif entry_instance_value != argument:
                    entry_matches = False
                    break
            if entry_matches:
                # m = len(table_indices)
                # n = len(cur_entry.table.shape)
                # for _ in range(n - m):
                #     table_indices.append(0)
                squeezed = np.squeeze(cur_entry.table)
                if len(squeezed.shape) == 0:
                    prob = float(squeezed)
                else:
                    prob = squeezed[tuple(table_indices)]
                return prob
        return 0.0

    def get_value(self, arguments):
        if len(arguments) == len(self.parent):
            return self._get_dist(arguments)
        elif len(arguments) == len(self.parent) + 1:
            return self._get_exact_prob(arguments)
        else:
            raise Exception("Invalid number of arguments")


class Action:
    def __init__(self, name):
        self.name = name


class Obs:
    def __init__(self, name):
        self.name = name


class State(dict):
    def __init__(self, vars_to_values):
        super().__init__(**vars_to_values)

    def __hash__(self):
        return hash(tuple((var_name, value) for var_name, value in self.items()))

    def __eq__(self, other):
        if len(self.keys()) != len(other.keys()):
            return False
        for k, v in self.items():
            if k not in other.keys() or other[k] != v:
                return False
        return True


class StateManager(ABC):
    @abstractmethod
    def create_state(self):
        pass


class FactoredStateManager(StateManager):
    def __init__(self, vars):
        self.vars = OrderedDict(**{var.name: var for var in vars})
        self.num_vars = len(vars)

    def create_state(self, varname_to_value=None, var_values=None):
        creation_dict = {}
        if varname_to_value is not None:
            for varname, value in varname_to_value.items():
                if varname not in self.vars:
                    raise Exception("Var given does not exist")
                var_object = self.vars[varname]
                if not var_object.is_valid_value(value):
                    raise Exception("Invalid value for var")
                creation_dict[varname] = value
        elif var_values is not None:
            if len(var_values) > self.num_vars:
                raise Exception("Number of values exceeds number of vars")

            vars_objects = list(self.vars.values())
            for idx, value in enumerate(var_values):
                if vars_objects[idx].is_valid_value(value):
                    creation_dict[self.vars[idx].name] = value
                else:
                    raise Exception("Invalid value for var")
        else:
            raise Exception("Must pass either varname to value or ordered values")
        return State(creation_dict)

    @staticmethod
    def create_empty_state():
        return State({})

    @staticmethod
    def update_state(state_1, state_2):
        if not isinstance(state_1, State) or not isinstance(state_2, State):
            raise Exception("one of the given arguments is not a state")
        for varname, value in state_2.items():
            state_1[varname] = value

    def update_state_var(self, state, varname, value):
        if varname in self.vars and value in self.vars[varname].domain:
            state[varname] = value


class Model(ABC):

    def __init__(self, initial_belief, state_manager, num_agents):
        self.initial_belief = initial_belief
        self.state_manager = state_manager
        self._num_agents = num_agents

    # def is_valid_joint_action(self, joint_action):
    #     if len(joint_action) != self.num_agents:
    #         return False
    #     for idx, action_name in enumerate(joint_action):
    #         available_actions = self.agent_actions[idx]
    #         if action_name not in available_actions:
    #             return False
    #     return True
    #
    # def is_valid_joint_obs(self, joint_obs):
    #     if len(joint_obs) != self.num_agents:
    #         return False
    #     for idx, obs_name in enumerate(joint_obs):
    #         available_obs = self.agent_obs[idx]
    #         if obs_name not in available_obs:
    #             return False
    #     return True

    @abstractmethod
    def sample_from_initial_state(self):
        pass

    @property
    @abstractmethod
    def states(self):
        return

    @property
    @abstractmethod
    def num_states(self):
        return

    @property
    @abstractmethod
    def num_actions(self):
        return

    @property
    @abstractmethod
    def num_observations(self):
        return

    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, num_agents):
        self._num_agents = num_agents

    @abstractmethod
    def step(self, cur_state, joint_action, random_tokens):
        """Returns next_state, reward and joint observation"""
        # next_state_dist = self.T(s=cur_state, ja=joint_action)
        # next_state = None
        # obs_dist = self.Z(s_tag=next_state, ja=joint_action)
        # joint_obs = None
        # reward = self.R(s=cur_state, ja=joint_action, s_tag=next_state)
        # return next_state, reward, joint_obs


class MatrixModel(Model):
    def T(self, s, ja, s_tag=None):
        pass

    def Z(self, s_tag, ja, jo=None):
        pass

    def R(self, s, ja, s_tag):
        pass


class PhaseOverseer:
    @property
    def prev(self):
        return "PAST"

    @property
    def current(self):
        return "CURRENT"

    @property
    def next(self):
        return "NEXT"

    def __init__(self, prev_phase_suffix, next_phase_suffix):
        self.prev_phase_suffix = prev_phase_suffix
        self.next_phase_suffix = next_phase_suffix

    def get_var_phase(self, varname):
        if varname.endswith(self.prev_phase_suffix):
            return self.prev
        if varname.endswith(self.next_phase_suffix):
            return self.next
        return self.current

    def get_unphased_var(self, varname):
        if varname.endswith(self.prev_phase_suffix):
            return varname[:-(len(self.prev_phase_suffix))]
        if varname.endswith(self.next_phase_suffix):
            return varname[:-(len(self.next_phase_suffix))]
        return varname


class POMDPXModel(Model):
    """Model derived from pomdpx that specifies central multi agent problem, with single action variable
    Prerequisits:
    - Single action transitions are described
    - No collborative sensing actions
    - Mapping of actions to actors
    - Mapping of each observation variable to its agent
    """
    state_vars = None  # String -> Var OrderedDict
    obs_vars = None  # String -> Var OrderedDict
    reward_vars = None  # String -> Var
    action_var = None  # Var

    transition_funcs = None  # String -> Func
    obs_funcs = None  # String -> Func
    reward_funcs = None  # String -> Func

    action_to_agent = None  # String -> tuple(int)
    obs_to_agent = None  # String -> tuple(int)

    initial_belief = None  # List(Dist)

    def __init__(self, state_vars, obs_vars, reward_vars, action_var,
                 transition_funcs, obs_funcs, reward_funcs,
                 action_to_agent, obs_to_agent, initial_belief, state_manager, num_agents,
                 phase_overseer):
        """
        :param state_vars: List(Var) - Order matters
        :param obs_vars: List(Var) - Order matters
        :param reward_vars: List(Var)
        :param action_var: Var
        :param transition_funcs: List(Var)
        :param obs_funcs: List(Var)
        :param reward_funcs: List(Var)
        :param action_to_agent: Dict<String, List(Integer)>
        :param obs_to_agent: Dict<String, Integer>
        """
        super().__init__(initial_belief=initial_belief, state_manager=state_manager, num_agents=num_agents)
        self.state_vars = OrderedDict(**{var.name: var for var in state_vars})
        self.obs_vars = OrderedDict(**{var.name: var for var in obs_vars})
        self.reward_vars = {var.name: var for var in reward_vars}
        self.action_var = action_var

        self.transition_funcs = {POMDPXMarks.clear_phase(func.var.name): func for func in transition_funcs}
        self.obs_funcs = {POMDPXMarks.clear_phase(func.var.name): func for func in obs_funcs}
        self.reward_funcs = {POMDPXMarks.clear_phase(func.var.name): func for func in reward_funcs}

        self.action_to_agent = action_to_agent
        self.obs_to_agent = obs_to_agent

        self.phase_overseer = phase_overseer
        for func in self.transition_funcs.values():
            func.create_instance = partial(func.create_instance, phase_overseer=self.phase_overseer)
        for func in self.obs_funcs.values():
            func.create_instance = partial(func.create_instance, phase_overseer=self.phase_overseer)
        for func in self.reward_funcs.values():
            func.create_instance = partial(func.create_instance, phase_overseer=self.phase_overseer)

    def sample_from_initial_state(self):
        result_state = self.state_manager.create_empty_state()
        for dist in self.initial_belief:
            partial_state = dist.sample()
            self.state_manager.update_state(result_state, partial_state)
        return result_state

    @property
    def states(self):
        return None

    @property
    def num_states(self):
        return reduce(lambda x, y: x * y, [len(var.domain) for var in self.state_vars.values()])

    @property
    def num_actions(self):
        return len(self.action_var.domain)

    @property
    def num_observations(self):
        return reduce(lambda x, y: x * y, [len(var.domain) for var in self.obs_vars.values()])

    @staticmethod
    def _detect_conflicts(origin_state, target_states):
        """Return list of pairs of agent indices that are in conflict
        Can be more efficient with indexing..."""
        conflicts = []
        for s1, s2 in itertools.combinations(target_states, 2):
            if s1 != origin_state and s2 != origin_state:
                diff_1 = {k: v for k, v in s1.items() if v != origin_state[k]}
                diff_2 = {k: v for k, v in s2.items() if v != origin_state[k]}

                joint_vars = set(diff_1.keys()).intersection(set(diff_2.keys()))
                states_conflicting = any([s1[var] != s2[var] for var in joint_vars])
                if states_conflicting:
                    conflicts.append((target_states.index(s1), target_states.index(s2)))

        return conflicts

    def _calculate_next_state(self, cur_state, joint_action, action_random_tokens):
        action_consumed = [False for _ in joint_action]
        resulting_states = []  # states emerging from applying each action group separately
        action_grouping_agents = []  # indices of the first agent in each action group
        consider_positive_reward = [True for _ in joint_action]

        for agent_idx, single_action in enumerate(joint_action):
            if action_consumed[agent_idx]:  # action was part of an already applied collaborative action
                continue

            cur_resulting_state = self.state_manager.create_empty_state()
            cur_random_token = action_random_tokens[agent_idx]

            # If one of the co-agents doesn't cooperate, we neutralize the action
            for co_agent_idx in self.action_to_agent[single_action]:
                if co_agent_idx != agent_idx:
                    if joint_action[co_agent_idx] != single_action:
                        cur_random_token = 0
                        consider_positive_reward[agent_idx] = False
                    else:
                        cur_random_token *= action_random_tokens[co_agent_idx]
                        action_consumed[co_agent_idx] = True

            # consume the agent's action and declare him as the leader of the action group
            action_consumed[agent_idx] = True
            action_grouping_agents.append(agent_idx)

            # sample the action group leader's resulting state
            for state_var in self.state_vars:
                t_func = self.transition_funcs[state_var]
                instance = t_func.create_instance(prev_vars=cur_state,
                                                  current_vars={self.action_var.name: single_action})

                if cur_random_token > 0:
                    sampled_var_value = t_func.get_value(instance).sample(cur_random_token)
                else:  # TODO not so good, if random token is 0 dont change. relevant only for deterministic dist
                    sampled_var_value = cur_state[state_var]
                self.state_manager.update_state_var(cur_resulting_state, state_var, sampled_var_value)

            resulting_states.append(cur_resulting_state)

        conflicts = [(action_grouping_agents[i], action_grouping_agents[j]) for i, j in
                     self._detect_conflicts(cur_state, resulting_states)]

        if len(conflicts) > 0:  # Conflicts detected
            eliminated_agents = []
            for conflict in conflicts:
                if not (conflict[0] in eliminated_agents or conflict[1] in eliminated_agents):
                    elimination_choice = 1 if random.uniform(0, 1) > 0.5 else 0  # randomly eliminate an agent
                    eliminated_agent = conflict[elimination_choice]
                    eliminated_agents.append(eliminated_agent)
                    action_random_tokens[eliminated_agent] = 0  # neutralize the agent's action

            # Eliminate current
            del conflicts
            del resulting_states
            del action_grouping_agents

            return self._calculate_next_state(cur_state,
                                              joint_action,
                                              action_random_tokens)  # All conflicting actions are eliminated

        else:  # No conflicts means we can safely merge all resulting states to obtain the next state
            next_state = self.state_manager.create_state(varname_to_value=cur_state)
            differential_updates = [self.state_manager.create_state(
                varname_to_value={k: v for k, v in r_state.items() if cur_state[k] != r_state[k]})
                for r_state in resulting_states]
            for update in differential_updates:
                self.state_manager.update_state(next_state, update)

        return next_state, consider_positive_reward

    def _calculate_all_obs(self, next_state, joint_action, obs_random_tokens):
        all_obs = []
        for obs_var in self.obs_vars:
            o_func = self.obs_funcs[obs_var]
            agent_idx = self.obs_to_agent[obs_var]
            single_action = joint_action[agent_idx]
            instance = o_func.create_instance(next_vars=next_state, current_vars={self.action_var.name: single_action})

            cur_random_token = obs_random_tokens[agent_idx]
            cur_obs = o_func.get_value(instance).sample(cur_random_token)
            all_obs.append(cur_obs)
        return all_obs

    def _calculate_reward(self, cur_state, joint_action, next_state, consider_positive_reward):
        res = 0.0
        for reward_var in self.reward_vars:
            used_instances = set()
            r_func = self.reward_funcs[reward_var]
            for idx, single_action in enumerate(joint_action):
                instance = tuple(r_func.create_instance(prev_vars=cur_state, next_vars=next_state,
                                                        current_vars={self.action_var.name: single_action}))

                # we reward according to the single action variable model, need to change it
                if instance not in used_instances:
                    curr_reward = r_func.get_value(instance)
                    res += curr_reward if curr_reward <= 0 or consider_positive_reward[idx] else 0.0
                    used_instances.add(instance)
        return res

    def step(self, cur_state, joint_action, random_tokens):
        """Returns next_state, reward and joint observation"""
        action_random_tokens, obs_random_tokens = [p[0] for p in random_tokens], [p[1] for p in random_tokens]
        next_state, consider_positive_reward = self._calculate_next_state(cur_state, joint_action, action_random_tokens)
        all_obs = self._calculate_all_obs(next_state, joint_action, obs_random_tokens)

        joint_obs = [[] for _ in range(self.num_agents)]
        sorted_obs_vars = list(self.obs_vars.keys())
        for idx, cur_obs in enumerate(all_obs):
            agent_idx = self.obs_to_agent[sorted_obs_vars[idx]]
            joint_obs[agent_idx].append(cur_obs)

        reward = self._calculate_reward(cur_state, joint_action, next_state, consider_positive_reward)

        return next_state, reward, joint_obs
