from abc import abstractmethod
from itertools import product
from weakref import WeakSet

import numpy as np
import re
from lxml import etree as ET
from collections import namedtuple

from POMDPX import POMDPXConstants
from POMDPX.POMDPXConstants import ProjectionMode, ITERATION_SIGN, WILDCARD_SIGN, FIXED_TABLE_VALUES_TO_CONSTRUCTORS, \
    FIXED_TABLE_VALUES


def create_element(*args, text=None, **kwargs):
    res = ET.Element(*args, **kwargs)
    if text:
        res.text = text
    res.tail = '\n'
    return res


class XMLParsable(object):
    @abstractmethod
    def get_element(self, name, two_sliced):
        pass


class Taggable(object):
    def __init__(self, tags):
        if type(tags) is str:
            tags = tags.split('_').strip()
        self.tags = tags


class Rewards:
    def __init__(self):
        pass


class Agent:
    def __init__(self):
        self.name = ""


class Action:
    def __init__(self):
        self.name = ""
        self.agents = []
        self.achievable_rewards = []


class Observation:
    def __init__(self):
        self.name = ""
        self.agent = ""


class Var(Taggable, XMLParsable):
    def __init__(self, name):
        super().__init__(tags=[name])
        self.name = name

    @property
    def prev(self):
        return self.name + '_0'

    @property
    def curr(self):
        return self.name + '_1'

    def get_element(self, name, two_sliced):
        attrib = dict()
        if two_sliced:
            attrib = {'vnamePrev': self.name + "_0",
                      'vnameCurr': self.name + "_1"
                      }
        else:
            attrib = {'vname': self.name}

        elem = create_element(name, attrib=attrib)

        return elem


class EnumVar(Var):
    def __init__(self, name, values):
        super().__init__(name)
        self.values = values

    def get_element(self, name, two_sliced):
        elem = super().get_element(name, two_sliced=two_sliced)
        values_enum = create_element('ValueEnum', text=' '.join(self.values))
        elem.append(values_enum)
        return elem


class Func(XMLParsable):
    def __init__(self, var, parents, entries):
        self.entries = entries
        self.parents = parents
        self.var = var
        self.subject = var.split('_')[-1].strip()  # TODO

    def get_element(self, name=None, is_prob=False):
        name = "CondProb" if is_prob else "Func"
        elem = create_element(name)

        var = create_element('Var', text=self.var)
        parent = create_element('Parent', text=' '.join(self.parents))
        elem.append(var)
        elem.append(parent)

        parameter = create_element('Parameter', attrib={'type': 'TBL'})
        for entry in self.entries:
            parameter.append(entry.get_element(is_prob=is_prob))
        elem.append(parameter)
        return elem


class Entry(Taggable, XMLParsable):
    def __init__(self, instance, table):
        super().__init__(tags=instance)
        self.instance = instance
        self.table = table

    def get_element(self, name=None, is_prob=False):
        elem = create_element("Entry")
        instance = create_element('Instance', text=' '.join(self.instance))
        table_string = POMDPXProblem.string_from_matrix(self.table) if not isinstance(self.table, str) else self.table
        prob_table = create_element('ProbTable' if is_prob else 'ValueTable',
                                    text=table_string)
        elem.append(instance)
        elem.append(prob_table)
        return elem

    def evaluate(self, indices):
        try:
            return np.squeeze(self.table[indices])
        except Exception() as e:
            return None


class POMDPXProblem:

    def __init__(self):
        # Inferred elements #
        self.utils = None
        self.agents = []
        self.num_agents = None
        self.actions = []
        self.time_zero_state_vars = []
        self.time_one_state_vars = []
        self.observations = []
        self.num_states = None
        self.subjects = {}
        self.agent_relevant_vars = {}

        # Explicit elements #

        # Variables
        self.state_vars = {}
        self.obs_vars = {}
        self.action_vars = {}
        self.reward_vars = {}

        # Initial Belief
        self.initial_belief = {}
        self.all_initial_states = {}

        # Functions
        self.state_transitions_func = {}
        self.observations_func = {}
        self.rewards_func = {}

        # Etc
        self.discount = 0
        self.description = ""

        #
        self.metadata = None
        self.use_metadata_for_projection = False

    def register_pomdpx_constants(self, pomdpx_constants):
        """Should eventually help to remove all pomdpxformat usages in the parses..."""
        pass

    def register_domain_constants(self, domain_constants):
        self.utils = domain_constants

    @staticmethod
    def is_wildcard_value(val):
        return val == WILDCARD_SIGN

    @staticmethod
    def is_iteration_value(val):
        return val == ITERATION_SIGN

    def is_exact_value(self, val):
        return not (self.is_wildcard_value(val) or self.is_iteration_value(val))

    def get_match_indices(self, searched_state, existing_state):
        vars = existing_state._fields
        candidate_vars = {**self.state_vars, **self.action_vars}
        indices = []
        for var in vars:
            existing_val = getattr(existing_state, var)
            if self.is_exact_value(existing_val):
                if getattr(searched_state, var) != existing_val:  # No match
                    return None
                continue
            elif self.is_wildcard_value(existing_val):
                continue
            else:  # Iteration value
                indices.append(candidate_vars[self.remove_time(var)].values.index(
                    getattr(searched_state, var)))  # TODO add centralized access to vars
        return tuple(indices)

    def calc_rewards(self, s_0: dict, a: str, s_1: dict):
        total_reward = 0.0
        rewards = []
        searched_state = {**{POMDPXConstants.time_zero(k): v for k, v in s_0.items()},
                          list(self.action_vars.keys())[0]: a,
                          **{POMDPXConstants.time_one(k): v for k, v in s_1.items()}}
        for reward_var in self.reward_vars:
            reward_func = self.rewards_func[reward_var]

            existing_state_vars = reward_func.parents.copy()
            CurrentFuncState = namedtuple(typename="CurrentFuncState", field_names=existing_state_vars)
            curr_reward = 0.0
            for entry in reversed(reward_func.entries):
                existing_state = CurrentFuncState(*entry.instance)
                narrowed_searched_state = CurrentFuncState(*[searched_state[var] for var in existing_state._fields])
                match_result = self.get_match_indices(narrowed_searched_state, existing_state)
                if match_result is not None:
                    squeezed_table = np.squeeze(entry.table)
                    if len(squeezed_table.shape) == 0:
                        curr_reward = float(squeezed_table)
                    else:
                        curr_reward = squeezed_table[match_result]
                    break
            if curr_reward != 0.0:
                rewards.append(curr_reward)
                total_reward += curr_reward
        return total_reward, rewards

    def calc_transition_prob(self, s_0: dict, a: str, s_1: dict):
        searched_state = {**{POMDPXConstants.time_zero(k): v for k, v in s_0.items()},
                          list(self.action_vars.keys())[0]: a,
                          **{POMDPXConstants.time_one(k): v for k, v in s_1.items()}}

        affected_vars = [v for v in s_0 if s_0[v] != s_1[v]]
        transition_prob = 1.0

        for affected_var in affected_vars:
            curr_prob = 1.0
            transition_func = self.state_transitions_func[POMDPXConstants.time_one(affected_var)]

            existing_state_vars = transition_func.parents.copy() + [POMDPXConstants.time_one(affected_var)]
            CurrentFuncState = namedtuple(typename="CurrentFuncState", field_names=existing_state_vars)
            for entry in reversed(transition_func.entries):
                existing_state = CurrentFuncState(*entry.instance)
                narrowed_searched_state = CurrentFuncState(*[searched_state[var] for var in existing_state._fields])
                match_result = self.get_match_indices(narrowed_searched_state, existing_state)
                if match_result is not None:
                    squeezed_table = np.squeeze(entry.table)
                    if len(squeezed_table.shape) == 0:
                        curr_prob = float(squeezed_table)
                    else:
                        curr_prob = squeezed_table[match_result]
                    break
            transition_prob *= curr_prob
        return transition_prob

    def parse_problem(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        self.description = self._parse_description(root.find('Description').text)
        self.discount = self._parse_discount(root.find('Discount').text)

        try:
            self.metadata = self.utils.parse_metadata(root.find('MetaData'))
            self.use_metadata_for_projection = True
        except Exception:
            print("Metadata not stated!")

        try:
            self.num_agents = self._parse_num_agents(root.find('NumAgents').text)
        except Exception:
            print("Num of agents not stated!")

        variables = root.find('Variable')
        state_vars = variables.findall('StateVar')
        obs_vars = variables.findall('ObsVar')
        action_vars = variables.findall('ActionVar')
        if len(action_vars) > 1:
            raise Exception("Only 1 action variable is allowed")
        reward_vars = variables.findall('RewardVar')

        for state_var in state_vars:
            self.handle_state_var(state_var)

        for obs_var in obs_vars:
            self.handle_obs_var(obs_var)

        for action_var in action_vars:
            self.handle_action_var(action_var)

        for reward_var in reward_vars:
            self.handle_reward_var(reward_var)

        initial_belief = root.find('InitialStateBelief')
        self.handle_initial_belief(initial_belief)

        state_transition_func = root.find('StateTransitionFunction')
        self.handle_state_transition_func(state_transition_func)

        obs_func = root.find('ObsFunction')
        if obs_func is not None:
            self.handle_obs_func(obs_func)

        reward_func = root.find('RewardFunction')
        self.handle_reward_func(reward_func)

        self.calculate_inferrables()

    def calculate_inferrables(self):
        for action_var in self.action_vars.values():
            for action in action_var.values:
                self.actions.append(action)

        if self.num_agents is not None:
            for i in range(self.num_agents):
                self.agents.append(self.utils.agent_symbol(i))

        for action in self.actions:
            self.subjects[action] = self.calculate_action_subjects(action)

        for agent in self.agents:
            self.agent_relevant_vars[agent] = self.calculate_agent_relevant_vars(agent)

        self.calc_all_initial_states()

    @staticmethod
    def matrix_from_string(matrix_string):
        rows = matrix_string.split('\n')
        return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])

    @staticmethod
    def string_from_matrix(matrix):
        rows_strings = [np.array2string(row, separator=' ', precision=2, formatter={
            'float_kind': lambda x: "%.4f" % x if x != 0.0 else "%d" % int(x)}).strip('[]') for row in matrix]
        return '\n'.join(rows_strings)

    def _construct_entry_from_entry_xml(self, xml_entry, is_prob, func_var_name=None):
        instance = xml_entry.find('Instance').text.strip().split()
        table_key = 'ProbTable' if is_prob else 'ValueTable'
        table_string = xml_entry.find(table_key).text.strip('\n ')

        if table_string in POMDPXConstants.FIXED_TABLE_VALUES:
            if table_string in FIXED_TABLE_VALUES:
                ctor = FIXED_TABLE_VALUES_TO_CONSTRUCTORS[table_string]
                n = len(self.state_vars[self.remove_time(func_var_name)].values)
                table = ctor(n)
            else:
                print("Unknow fixed table value %s" % table_string)
                table = table_string
        else:
            table = self.matrix_from_string(table_string)
        return Entry(instance=instance, table=table)

    def _construct_func_from_func_xml(self, xml_func, is_condprob):
        var = xml_func.find('Var').text
        parents = xml_func.find('Parent').text.strip().split()
        entries = []

        for entry_xml in xml_func.find('Parameter'):
            entries.append(self._construct_entry_from_entry_xml(entry_xml, is_prob=is_condprob, func_var_name=var))

        return Func(var=var, parents=parents, entries=entries)

    def handle_state_var(self, xml_state_var):
        name = self._parse_state_var_prev_name(xml_state_var.attrib['vnamePrev'])
        values = self._parse_var_value_enum(xml_state_var.find('ValueEnum').text)
        state_var = EnumVar(name, values)
        self.state_vars[name] = state_var
        self.time_zero_state_vars.append(state_var.prev)
        self.time_one_state_vars.append(state_var.curr)

    def handle_obs_var(self, xml_obs_var):
        name = self._parse_obs_var_name(xml_obs_var.attrib['vname'])
        values = self._parse_var_value_enum(xml_obs_var.find('ValueEnum').text)
        obs_var = EnumVar(name, values)
        self.obs_vars[name] = obs_var

    def handle_action_var(self, xml_action_var):
        name = self._parse_action_var_name(xml_action_var.attrib['vname'])
        values = self._parse_var_value_enum(xml_action_var.find('ValueEnum').text)
        action_var = EnumVar(name, values)
        self.action_vars[name] = action_var

    def add_reward_var(self, name):
        reward_var = Var(name)
        self.reward_vars[name] = reward_var

    def is_reward_var(self, var_name):
        return var_name in self.reward_vars

    def is_action_var(self, var_name):
        return var_name in self.action_vars

    def is_state_var(self, var_name):
        return self.is_prev_state_var(var_name) or self.is_curr_state_var(var_name)

    def is_prev_state_var(self, var_name):
        return var_name in self.time_zero_state_vars

    def is_curr_state_var(self, var_name):
        return var_name in self.time_one_state_vars

    def add_reward_func(self, var, parents_list):
        if not self.is_reward_var(var):
            raise Exception("%s is not a reward variable" % var)
        for parent_var in parents_list:
            if not self.is_action_var(parent_var) and \
                    not self.is_curr_state_var(parent_var) and \
                    not self.is_prev_state_var(parent_var):
                raise Exception("%s is neither an action var nor a timed state var" % parent_var)

        self.rewards_func[var] = Func(var=var, parents=parents_list, entries=[])

    def get_enum_var_object(self, name):
        if self.is_action_var(name):
            return self.action_vars[name]
        elif name in self.obs_vars:
            return self.obs_vars[name]
        elif self.is_state_var(name):
            for state_var in self.state_vars.values():
                if name == state_var.prev or name == state_var.curr:
                    return state_var
        else:
            raise Exception("Var named %s does not exist" % name)

    def _validate_instance_with_var(self, instance, var):
        var = self.get_enum_var_object(name=var)
        if instance not in var.values:
            raise Exception("%s is not a valid value of var %s" % (instance, var))

    def _validate_instance_list_with_parent_list(self, instance_list, parent_list):
        num_instances = len(instance_list)
        num_parents = len(parent_list)
        if num_instances != num_parents:
            raise Exception("Number of parents must be equal to number of instances components")
        for instance_index, instance_component in enumerate(instance_list):
            if instance_component == POMDPXConstants.ITERATION_SIGN and instance_index < num_instances - 2:
                raise Exception("Iteration token can be applied only to last two instances")
            elif instance_component == POMDPXConstants.WILDCARD_SIGN:
                continue
            else:
                self._validate_instance_with_var(instance_component, parent_list[instance_index])

    def add_entry_to_func(self, func, instance_string, table):
        instance_list = instance_string.strip(' \n').split()
        if len(table.shape) == 0:  # single value
            table = np.asmatrix(table)
        self._validate_instance_list_with_parent_list(instance_list=instance_list,
                                                      parent_list=func.parents)

        entry = Entry(instance=instance_list, table=table)
        func.entries.append(entry)

    def handle_reward_var(self, xml_reward_var):
        name = self._parse_action_var_name(xml_reward_var.attrib['vname'])
        self.add_reward_var(name)

    def handle_initial_belief(self, xml_initial_belief):
        var_to_dist = {}
        for cond_prob in xml_initial_belief.findall('CondProb'):
            cond_prob = self._construct_func_from_func_xml(cond_prob, is_condprob=True)
            cur_var_name = cond_prob.var
            var_to_dist[cur_var_name] = cond_prob
        self.initial_belief = var_to_dist

    def calc_all_initial_states(self):
        factored_initial_values = []
        for var, dist in self.initial_belief.items():
            valid_values = set()
            for cur_entry in dist.entries:
                if cur_entry.table == "uniform":
                    valid_values = valid_values.union(set(self.state_vars[POMDPXConstants.clear_phase(var)].values))
                else:
                    valid_values = valid_values.union(
                        set([cur_entry.instance[idx] for idx in range(len(cur_entry.instance)) if
                             np.asmatrix(cur_entry.table)[0, idx] > 0]))
            factored_initial_values.append(valid_values)

        self.all_initial_states = set(product(*factored_initial_values))

    def handle_state_transition_func(self, xml_state_transition_func):
        var_to_dist = {}
        for cond_prob in xml_state_transition_func.findall('CondProb'):
            cond_prob = self._construct_func_from_func_xml(cond_prob, is_condprob=True)
            cur_var_name = cond_prob.var.strip()
            var_to_dist[cur_var_name] = cond_prob
        self.state_transitions_func = var_to_dist

    def handle_obs_func(self, xml_obs_func):
        var_to_dist = {}
        for cond_prob in xml_obs_func.findall('CondProb'):
            cond_prob = self._construct_func_from_func_xml(cond_prob, is_condprob=True)
            cur_var_name = cond_prob.var.strip()
            var_to_dist[cur_var_name] = cond_prob
        self.observations_func = var_to_dist

    def handle_reward_func(self, xml_reward_func):
        var_to_rewards = {}
        for func in xml_reward_func.findall('Func'):
            func = self._construct_func_from_func_xml(func, is_condprob=False)
            cur_var_name = func.var.strip()
            var_to_rewards[cur_var_name] = func
        self.rewards_func = var_to_rewards

    def create_initial_belief_element(self):
        res = create_element('InitialStateBelief')
        for condprob in self.initial_belief.values():
            res.append(condprob.get_element(is_prob=True))
        return res

    def create_state_transition_func_element(self):
        res = create_element('StateTransitionFunction')
        for condprob in self.state_transitions_func.values():
            res.append(condprob.get_element(is_prob=True))
        return res

    def create_obs_func_element(self):
        res = create_element('ObsFunction')
        for condprob in self.observations_func.values():
            res.append(condprob.get_element(is_prob=True))
        return res

    def create_reward_func_element(self):
        res = create_element('RewardFunction')
        for func in self.rewards_func.values():
            res.append(func.get_element(is_prob=False))
        return res

    def delete_action(self, action_name):
        """
        Assumes no '-' iterates over an action variable!
        """
        self.actions.remove(action_name)

        containing_action_vars = set()
        for action_var in self.action_vars.values():
            action_var.values.remove(action_name)
            containing_action_vars.add(action_var.name)

        for func in [*self.state_transitions_func.values(),
                     *self.observations_func.values(),
                     *self.rewards_func.values()]:
            if not set(func.parents).isdisjoint(containing_action_vars):
                entries_to_remove = []
                for entry in func.entries:
                    if action_name in entry.instance:
                        entries_to_remove.append(entry)

                for entry in entries_to_remove:
                    func.entries.remove(entry)

    def delete_obs_var(self, obs_var):
        self.obs_vars.pop(obs_var)
        self.observations_func.pop(obs_var)

    def _parse_description(self, text):
        return text

    def _parse_discount(self, text):
        return float(text)

    def _parse_num_agents(self, text):
        return int(text)

    def _parse_state_var_prev_name(self, text):
        return text[:text.rfind('_')].strip()

    def _parse_obs_var_name(self, text):
        return text.strip()

    def _parse_action_var_name(self, text):
        return text.strip()

    def _parse_var_value_enum(self, text):
        return text.strip(' ').split()

    def write_problem(self, path):
        root = ET.Element('pomdpx')

        description = create_element('Description', text=str(self.description))
        root.append(description)

        discount = create_element('Discount', text=str(self.discount))
        root.append(discount)

        variables = create_element('Variable')

        for state_var in self.state_vars.values():
            variables.append(state_var.get_element(name='StateVar', two_sliced=True))

        for obs_var in self.obs_vars.values():
            variables.append(obs_var.get_element(name='ObsVar', two_sliced=False))

        for action_var in self.action_vars.values():
            variables.append(action_var.get_element(name='ActionVar', two_sliced=False))

        for reward_var in self.reward_vars.values():
            variables.append(reward_var.get_element(name='RewardVar', two_sliced=False))

        root.append(variables)

        initial_belief = self.create_initial_belief_element()
        root.append(initial_belief)

        state_transition_func = self.create_state_transition_func_element()
        root.append(state_transition_func)

        obs_func = self.create_obs_func_element()
        root.append(obs_func)

        reward_func = self.create_reward_func_element()
        root.append(reward_func)

        tree = ET.ElementTree(root)

        tree.write(path)
        with open(path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(POMDPXConstants.XML_HEADER + "\n" + content)

    def determinize_action(self, action_name):
        containing_action_vars = set()
        for action_var in self.action_vars.values():
            if action_name in action_var.values:
                containing_action_vars.add(action_var.name)

        for func in [*self.state_transitions_func.values(),
                     *self.observations_func.values()]:
            if not set(func.parents).isdisjoint(containing_action_vars):
                entries_to_flatten = []
                for entry in func.entries:
                    if action_name in entry.instance:
                        entries_to_flatten.append(entry)

                for entry in entries_to_flatten:
                    for idx, row in enumerate(entry.table):
                        max_prob_index = np.argmax(row)
                        entry.table[idx] = np.zeros(len(row))
                        entry.table[idx, max_prob_index] = 1.0

    def calculate_agent_relevant_vars(self, agent):
        """For now we define relevant as union of objectives of actions"""
        res = set()
        for action in self.actions:
            if agent in self.utils.extract_agents_from_action(action):
                res = res.union(set(self.utils.extract_objectives_from_action(action, self.metadata)))
        return res

    def calculate_action_subjects(self, action, from_constants=True):
        if from_constants:
            # if self.utils.is_collaborative_action(action):
            #     res = set(self.utils.extract_objectives_from_action(action))
            # else:
            #     res = {*self.utils.extract_objectives_from_action(action),
            #            *self.utils.extract_preconditions_from_action(action)}
            res = {*self.utils.extract_objectives_from_action(action, self.metadata),
                   *self.utils.extract_preconditions_from_action(action, self.metadata)}
        else:
            res = set()
            explicit_sources = [self.state_transitions_func, self.observations_func]
            implicit_sources = [self.reward_vars]
            subjects = []

            for var_source in explicit_sources:
                for var, func in var_source.items():
                    parents = func.parents.split()
                    if set(self.action_vars).isdisjoint(set(func.parents)):  # No action variable specified in func
                        raise Exception("explicit source doesn't contain action var")
                    try:
                        affected_var = self.state_vars[func.subject]
                    except KeyError:
                        affected_var = self.obs_vars[func.subject]
                    # meaning we need to action to be stated explicitly
                    related_entries = [entry for entry in reversed(func.entries) if action in entry.instance]
                    if affected_var.prev in parents:  # possibly an objective, if change occurs with prob > 0
                        is_objective = False
                        for related_entry in related_entries:
                            curr_instance = related_entry.instance.split()
                            time_zero_idx = parents.index(affected_var.prev)
                            prev_value = curr_instance[time_zero_idx]
                            curr_value = curr_instance[-1]
                            if prev_value == ITERATION_SIGN:
                                prev_value = affected_var.values
                            else:
                                prev_value = [prev_value]
                            if curr_value == ITERATION_SIGN:
                                curr_value = affected_var.values
                            else:
                                curr_value = [curr_value]
                            table = related_entry.table
                            for p_idx, p_val in enumerate(prev_value):
                                for c_idx, c_val in enumerate(curr_value):
                                    if p_val != c_val and table[p_idx][c_idx] > 0:
                                        is_objective = True
                            if is_objective:
                                subjects.append(affected_var.name)
                                break
                    # TODO handle preconditions
        return res

    def project_instance(self, parents, state_elements, action):
        res = []
        state_idx = 0
        if self.use_metadata_for_projection:
            action_objectives = self.utils.extract_objectives_from_action(action, self.metadata)
        else:
            action_objectives = self.utils.extract_objectives_from_action(action)
        action_agents = self.utils.extract_agents_from_action(action)

        for parent in parents:
            if parent in self.action_vars:
                res.append(action)
            else:  # parent is a state var
                should_project = True
                for agent in action_agents:
                    if agent in parent:  # TODO, inconsistent with projection denfinitions
                        should_project = False
                        break
                for subject in action_objectives:
                    if subject in parent:  # TODO
                        should_project = False
                        break

                if should_project:
                    res.append('*')
                else:
                    res.append(state_elements[state_idx])
                state_idx += 1
        return res

    def project_instance_v2(self, parents, state_elements, action, agent=None,
                            mode=ProjectionMode.RELEVANT_PLUS_SUBJECTS):
        res = []
        state_idx = 0
        action_subjects = self.subjects[action]
        agent_relevant_vars = self.agent_relevant_vars[agent] if agent is not None else None
        projection_variables = set()
        for action_var in self.action_vars:
            projection_variables.add(action_var)
        if mode == ProjectionMode.SUBJECTS:
            projection_variables = projection_variables.union(action_subjects)
        elif mode == ProjectionMode.RELEVANT_VARS:
            projection_variables = projection_variables.union(agent_relevant_vars)
        elif mode == ProjectionMode.RELEVANT_SUBJECTS:
            projection_variables = projection_variables.union(agent_relevant_vars.intersection(action_subjects))
        elif mode == ProjectionMode.RELEVANT_PLUS_SUBJECTS:
            projection_variables = projection_variables.union(agent_relevant_vars.union(action_subjects))
        else:
            raise Exception("Unsupported projection mode")

        for parent in parents:
            parent = self.remove_time(parent)
            should_project = parent in projection_variables  # TODO check timestamp
            if not should_project:
                res.append('*')
                state_idx += 1
            elif parent not in self.action_vars:
                res.append(state_elements[state_idx])
                state_idx += 1
            else:
                res.append(action)
        return res

    def remove_time(self, var_name):
        if var_name.endswith('_1') or var_name.endswith('_0'):
            return var_name[:-2]
        return var_name
