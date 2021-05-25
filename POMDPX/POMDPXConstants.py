from typing import Callable, List
import re
import numpy as np
from enum import Enum

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
DECLARE_NUMAGENTS = 'NumAgents'
DECLARE_METADATA = 'MetaData'

FIELD_VAR = 'Var'
FIELD_PARENT = 'Parent'
FIELD_INSTANCE = 'Instance'
FIELD_PARAMETER = 'Parameter'
FIELD_VALUE_TABLE = 'ValueTable'
FIELD_PROB_TABLE = 'ProbTable'

FIXED_TABLE_VALUES_TO_CONSTRUCTORS = {'identity': lambda n: np.eye(n),
                                      'uniform': lambda n: np.ones(shape=(1, n)) / n}

PHASE_PREV = '_0'
PHASE_NEXT = '_1'
PREV_ATTRIBUTE = 'vnamePrev'
NEXT_ATTRIBUTE = 'vnameCurr'
CURRENT_ATTRIBUTE = 'vname'

NULL_PARENT = 'null'

XML_HEADER = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"
FIXED_TABLE_VALUES = ['identity', 'uniform']  # TODO from POMDPXConstants


# Projection Schemes
class ProjectionMode(Enum):
    SUBJECTS = 1
    RELEVANT_VARS = 2
    RELEVANT_SUBJECTS = 3
    RELEVANT_PLUS_SUBJECTS = 4


def clear_phase(var_text):
    if var_text.endswith(PHASE_NEXT):
        return var_text[:-(len(PHASE_NEXT))]
    elif var_text.endswith(PHASE_PREV):
        return var_text[:-(len(PHASE_PREV))]
    return var_text


def time_zero(var_name):
    return clear_phase(var_name) + PHASE_PREV


def time_one(var_name):
    return clear_phase(var_name) + PHASE_NEXT
