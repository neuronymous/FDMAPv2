from abc import ABC, abstractmethod
from typing import List, Callable

from lxml import etree as ET
import re

import numpy as np

from DecPOMDPSimulator.POMDPXMarks import POMDPXMarks
from DecPOMDPSimulator.Distribution import Distribution
from DecPOMDPSimulator.DecPOMDPModel import POMDPXModel, RealNumbers, Var, FactoredStateManager, Func, PhaseOverseer, \
    ProbFunc

from POMDPX.POMDPXFactory import POMDPXProblemFactory


class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, *args):
        pass


class POMDPXModelFactory(ModelFactory):
    default_domain = RealNumbers
    all_vars = None

    def create_model(self, pomdpx_path,
                     problem_kind,
                     num_agents=None):
        tree = ET.parse(pomdpx_path)
        root = tree.getroot()

        problem_constants = POMDPXProblemFactory.get_problem_constants_by_kind(problem_kind)
        extract_actors = problem_constants.extract_agents_from_action
        obs_var_to_agent_func = problem_constants.extract_agent_from_obsvar
        agent_symbol_maker = problem_constants.agent_symbol

        variables_xml = root.find(POMDPXMarks.DECLARE_VARIABLES)
        state_vars_xml = variables_xml.findall(POMDPXMarks.DECLARE_STATEVAR)
        obs_vars_xml = variables_xml.findall(POMDPXMarks.DECLARE_OBSVAR)
        action_vars_xml = variables_xml.findall(POMDPXMarks.DECLARE_ACTIONVAR)
        if len(action_vars_xml) > 1:
            raise Exception("Only 1 action variable is allowed")
        reward_vars_xml = variables_xml.findall(POMDPXMarks.DECLARE_REWARDVAR)

        state_vars = []
        obs_vars = []
        reward_vars = []

        for state_var_xml in state_vars_xml:
            state_vars.append(self.var_from_xml(state_var_xml))

        for obs_var_xml in obs_vars_xml:
            obs_vars.append(self.var_from_xml(obs_var_xml))

        for reward_var_xml in reward_vars_xml:
            reward_vars.append(self.var_from_xml(reward_var_xml))

        action_var = self.var_from_xml(action_vars_xml[0])

        self.all_vars = {**{var.name: var for var in state_vars},
                         **{var.name: var for var in obs_vars},
                         **{var.name: var for var in reward_vars},
                         action_var.name: action_var}

        state_manager = FactoredStateManager(state_vars)
        phase_overseer = PhaseOverseer(POMDPXMarks.PHASE_PREV, POMDPXMarks.PHASE_NEXT)

        initial_belief_xml = root.find(POMDPXMarks.DECLARE_INITIALBELIEF)
        var_to_initial_func = self.initial_belief_from_xml(initial_belief_xml)
        initial_belief = []
        for var_name, func in var_to_initial_func.items():
            cur_dist = {}
            for domain_value in self.all_vars[var_name].domain:
                partial_state = state_manager.create_state(varname_to_value={var_name: domain_value})
                prob = func.get_value(
                    func.create_instance(phase_overseer=phase_overseer, prev_vars=partial_state))
                cur_dist[partial_state] = prob
            initial_belief.append(Distribution(cur_dist))

        transition_funcs_xml = root.find(POMDPXMarks.DECLARE_TRANSITIONFUNC)
        transition_funcs = self.funcs_from_xml(transition_funcs_xml, is_prob_table=True)

        obs_funcs_xml = root.find(POMDPXMarks.DECLARE_OBSFUNC)
        obs_funcs = self.funcs_from_xml(obs_funcs_xml, is_prob_table=True)

        reward_funcs_xml = root.find(POMDPXMarks.DECLARE_REWARDFUNC)
        reward_funcs = self.funcs_from_xml(reward_funcs_xml, is_prob_table=False)

        # can be inferred only if agent symbols are sequential: a1,...a{num_agents}
        if num_agents is None:
            agents_found = set()
            agents_per_action = [extract_actors(action) for action in action_var.domain]
            for agents_bunch in agents_per_action:
                for agent in agents_bunch:
                    agents_found.add(agent)
            num_agents = len(agents_found)

        agents = [agent_symbol_maker(i) for i in range(num_agents)]
        action_to_agent = {action_name: [agents.index(agent) for agent in extract_actors(action_name)] for action_name
                           in action_var.domain}
        obs_to_agent = {var.name: agents.index(obs_var_to_agent_func(var.name)) for var in obs_vars}

        return POMDPXModel(state_vars=state_vars, obs_vars=obs_vars, reward_vars=reward_vars, action_var=action_var,
                           transition_funcs=transition_funcs, obs_funcs=obs_funcs, reward_funcs=reward_funcs,
                           state_manager=state_manager, initial_belief=initial_belief,
                           action_to_agent=action_to_agent, obs_to_agent=obs_to_agent,
                           num_agents=num_agents, phase_overseer=phase_overseer)

    def initial_belief_from_xml(self, xml_initial_belief):
        var_to_func = {}
        for func_xml in xml_initial_belief.findall(POMDPXMarks.DECLARE_CONDPROB):
            func = self._construct_func_from_func_xml(func_xml, is_prob_table=True)
            cur_var_name = POMDPXMarks.clear_phase(func.var.name)
            var_to_func[cur_var_name] = func
        return var_to_func

    def funcs_from_xml(self, funcs_xml, is_prob_table):
        funcs = []
        func_mark = POMDPXMarks.DECLARE_CONDPROB if is_prob_table else POMDPXMarks.DECLARE_RFUNC
        for func_xml in funcs_xml.findall(func_mark):
            func = self._construct_func_from_func_xml(func_xml, is_prob_table=is_prob_table)
            funcs.append(func)
        return funcs

    def _construct_func_from_func_xml(self, xml_func, is_prob_table):
        phased_var_name = xml_func.find(POMDPXMarks.FIELD_VAR).text.strip()
        var_name = POMDPXMarks.clear_phase(phased_var_name)
        phased_parents_names = list(filter(lambda name: name != POMDPXMarks.NULL_PARENT,
                                           xml_func.find(POMDPXMarks.FIELD_PARENT).text.strip().split()))
        parent = []
        for parent_name in phased_parents_names:
            domain = self.all_vars[POMDPXMarks.clear_phase(parent_name)].domain
            parent.append(Var(name=parent_name, values=domain))

        var = Var(name=phased_var_name, values=self.all_vars[var_name].domain)

        f = ProbFunc(var=var, parent=parent) if is_prob_table else Func(var=var, parent=parent)

        for entry_xml in xml_func.find(POMDPXMarks.FIELD_PARAMETER):
            instance = entry_xml.find(POMDPXMarks.FIELD_INSTANCE).text.strip().split()
            table_key = POMDPXMarks.FIELD_PROB_TABLE if is_prob_table else POMDPXMarks.FIELD_VALUE_TABLE
            table_string = entry_xml.find(table_key).text.strip('\n ')

            if table_string in POMDPXMarks.FIXED_TABLE_VALUES_TO_CONSTRUCTORS.keys():
                table = POMDPXMarks.FIXED_TABLE_VALUES_TO_CONSTRUCTORS[table_string](len(var.domain))
            else:
                table = self._matrix_from_string(table_string)
            f.create_and_add_entry(instance=instance, table=table)
        return f

    def var_from_xml(self, var_xml):
        try:
            var_name = POMDPXMarks.clear_phase(var_xml.attrib[POMDPXMarks.PREV_ATTRIBUTE])
        except Exception as e:
            var_name = var_xml.attrib[POMDPXMarks.CURRENT_ATTRIBUTE]
        try:
            values = self._parse_var_value_enum(var_xml.find(POMDPXMarks.DECLARE_VALUE_ENUM).text)
        except Exception as e:
            values = self.default_domain

        return Var(var_name, values)

    @staticmethod
    def _parse_var_value_enum(text):
        return text.strip(' ').split()

    @staticmethod
    def _matrix_from_string(matrix_string):
        rows = matrix_string.split('\n')
        return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
