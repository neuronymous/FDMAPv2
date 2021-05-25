from typing import Callable

NULL_OBS = 'null'
BUTTON_STATES = ['on', 'off', 'fail']
# Metadata:
# RocksLocations: List(int)
# ControlAreas: List(ControlArea: List(int))


WILDCARD = '*'


def obs_var():
    return 'o_null'


def agent_symbol(agent_idx):
    return 'agent%d' % (agent_idx + 1)


def button_symbol():
    return 'button'


def agent_cost_var(agent_idx):
    return 'cost_%s' % agent_symbol(agent_idx)


def button_reward_var():
    return 'reward_button'


def action_var():
    return 'main_action_var'


def move_action(agent_idx):
    return 'action_move_%s' % agent_symbol(agent_idx)


def click_action(*agents):
    return 'action_click_' + '_'.join([agent_symbol(agent_idx) for agent_idx in agents])


def idle_action():
    return 'action_idle'


def button_on():
    return BUTTON_STATES[0]


def button_off():
    return BUTTON_STATES[1]


def button_fail():
    return BUTTON_STATES[2]


def get_controller_from_action(action):
    return action.split('_')[-1]


def get_agent_from_action(action):
    return get_controller_from_action(action)


def get_controller_idx_from_symbol(controller_symbol):
    return int(controller_symbol[len('controller'):]) - 1


def is_public_action(action):
    return action.startswith('action_click')


def is_sense_action(action):
    return False


def is_idle_action(action):
    return action == idle_action()


def is_collaborative_action(action):
    return False


def extract_agents_from_action(action):
    """Note that agents are actually the controllers, as they are the action deciders"""
    return [comp for comp in action.split('_') if comp.startswith('agent')]


def extract_objectives_from_action(action, metadata=None):
    if action.startswith('action_move'):  # move
        objectives = [comp for comp in action.split('_') if comp.startswith('agent')]
    else:  # click
        objectives = [button_symbol()]
    return objectives


def extract_agents_from_obsvar(obsvar):
    return []


def extract_agent_from_obsvar(obsvar):
    return None


def parse_metadata(metadata_xml_elem):
    pass


def extract_metadata_objectives_from_action(action, metadata):
    pass
