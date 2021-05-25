from typing import Callable, List
import re
import numpy as np


class POMDPXMarks:
    """Contains all the information that should be enforced by the generator"""
    WILDCARD_SIGN = '*'
    ITERATION_SIGN = '-'
    DECLARE_CONDPROB = 'CondProb'
    DECLARE_RFUNC = 'Func'
    DECLARE_VALUE_ENUM = 'ValueEnum'
    DECLARE_VARIABLES = 'Variable'
    DECLARE_STATEVAR = 'StateVar'
    DECLARE_OBSVAR = 'ObsVar'
    DECLARE_ACTIONVAR = 'ActionVar'
    DECLARE_REWARDVAR = 'RewardVar'
    DECLARE_INITIALBELIEF = 'InitialStateBelief'
    DECLARE_TRANSITIONFUNC = 'StateTransitionFunction'
    DECLARE_OBSFUNC = 'ObsFunction'
    DECLARE_REWARDFUNC = 'RewardFunction'

    FIELD_VAR = 'Var'
    FIELD_PARENT = 'Parent'
    FIELD_INSTANCE = 'Instance'
    FIELD_PARAMETER = 'Parameter'
    FIELD_VALUE_TABLE = 'ValueTable'
    FIELD_PROB_TABLE = 'ProbTable'

    FIXED_TABLE_VALUES_TO_CONSTRUCTORS = {'identity': lambda n: np.eye(n)}

    PHASE_PREV = '_0'
    PHASE_NEXT = '_1'
    PREV_ATTRIBUTE = 'vnamePrev'
    NEXT_ATTRIBUTE = 'vnameCurr'
    CURRENT_ATTRIBUTE = 'vname'
    AGENT_REGEX = 'a\d'
    NULL_PARENT = 'null'
    IDLE_ACTION = 'aidle'
    extract_actors: Callable[[str], List[str]] = lambda x: [comp for comp in x.split('_') if
                                                            re.match(POMDPXMarks.AGENT_REGEX, comp)]
    OBS_VAR_TO_AGENT_FUNC: Callable[[str], str] = lambda x: POMDPXMarks.extract_actors(x)[-1]
    AGENT_SYMBOL_MAKER: Callable[[int], str] = lambda idx: "a%d" % (idx + 1)

    @staticmethod
    def clear_phase(var_text):
        if var_text.endswith(POMDPXMarks.PHASE_NEXT):
            return var_text[:-(len(POMDPXMarks.PHASE_NEXT))]
        elif var_text.endswith(POMDPXMarks.PHASE_PREV):
            return var_text[:-(len(POMDPXMarks.PHASE_PREV))]
        return var_text

    @staticmethod
    def is_public_action(action_name):
        return action_name.startswith('ap') or action_name.startswith('ajp')

    @staticmethod
    def is_collaborative_action(action_name):
        return action_name.startswith('ajp')
