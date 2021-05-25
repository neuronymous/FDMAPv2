from typing import Callable

DIRECTION_SYMBOLS = ['left', 'right']
NULL_OBS = 'null'
YES_OBS = 'yes'
NO_OBS = 'no'
OBS_VALUES = [YES_OBS, NO_OBS, NULL_OBS]
PROBLEM_NAME = "DT"

# Metadata:
# RocksLocations: List(int)
# ControlAreas: List(ControlArea: List(int))

left: Callable[[], str] = lambda: DIRECTION_SYMBOLS[0]
right: Callable[[], str] = lambda: DIRECTION_SYMBOLS[1]
up: Callable[[], str] = lambda: DIRECTION_SYMBOLS[2]
down: Callable[[], str] = lambda: DIRECTION_SYMBOLS[3]
WILDCARD = '*'
yes_idx = 0
no_idx = 1
null_idx = 2

def agent_symbol(agent_idx):
    return 'agent%d' % (agent_idx + 1)


def tiger_symbol():
    return 'tiger'


def obs_symbol(agent):
    return 'obs_%s' % (agent)


def agent_rminus_var(agent_idx):
    return 'rminus_%s' % agent_symbol(agent_idx)


def tiger_reward_variable():
    return 'reward_tiger'


def action_var():
    return 'action_var'


def move_action(direction_symbol, agent_idx):
    return 'move_%s_%s' % (direction_symbol, agent_symbol(agent_idx))


def open_action(agent_idx):
    return 'open_%s' % (agent_symbol(agent_idx))


def copen_action(*agents_indices):
    return 'copen_%s' % '_'.join([agent_symbol(agent_idx) for agent_idx in agents_indices])


def listen_action(agent_idx):
    return 'listen_%s' % (agent_symbol(agent_idx))


def idle_action():
    return 'idle'


def get_agent_idx_from_symbol(agent_symbol):
    return int(agent_symbol[len('agent'):]) - 1


def is_public_action(action):
    return is_open_action(action) or is_copen_action(action)


def is_open_action(action):
    return action.startswith('open')


def is_move_action(action):
    return action.startswith('move')


def is_copen_action(action):
    return action.startswith('copen')


def is_listen_action(action):
    return action.startswith('listen')


def is_idle_action(action):
    return action == idle_action()


def is_sense_action(action):
    return is_listen_action(action)


def is_collaborative_action(action):
    return is_copen_action(action)


def extract_agents_from_action(action):
    return [comp for comp in action.split('_') if comp.startswith('agent')]


def extract_agents_from_obsvar(obsvar):
    return [comp for comp in obsvar.split('_') if comp.startswith('agent')]


def extract_agent_from_obsvar(obsvar):
    agents = extract_agents_from_obsvar(obsvar)
    if len(agents) != 1:
        raise Exception("Expected exactly one agent")
    return agents[0]


def get_direction(action):
    if not is_move_action(action):
        raise Exception("Only move action contains direction")
    for d in DIRECTION_SYMBOLS:
        if d in action:
            return d
    return None


def extract_objectives_from_action(action, metadata=None):
    if is_public_action(action):
        return [tiger_symbol()]
    elif is_idle_action(action):
        return []
    else:
        agents = extract_agents_from_action(action)
        if is_listen_action(action):
            return [obs_symbol(agents[0])]
        else:  # move
            return agents


def extract_preconditions_from_action(action):
    if is_open_action(action) or is_copen_action(action) or is_listen_action(action):
        return [*extract_agents_from_action(action), tiger_symbol()]
    elif is_idle_action(action):
        return []
    elif is_move_action(action):
        return extract_agents_from_action(action)


def state_measure(state):
    return 0


def is_final_state(state):
    return False


def is_empty_state(state):
    return False

