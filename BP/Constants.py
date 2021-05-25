from typing import Callable

DIRECTION_SYMBOLS = ['left', 'right', 'up', 'down']
NULL_OBS = 'null'
YES_OBS = 'yes'
NO_OBS = 'no'
OBS_VALUES = [YES_OBS, NO_OBS, NULL_OBS]

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


def box_symbol(box_idx):
    return 'box%d' % (box_idx + 1)


def obs_symbol(box_idx, agent_idx):
    return 'obs_%s_%s' % (box_symbol(box_idx), agent_symbol(agent_idx))


def minobs_symbol(box):
    if type(box) is int:
        return 'obs_%s' % box_symbol(box)
    else:
        return 'obs_%s' % box


def agent_cost_var(agent_idx):
    return 'cost_%s' % agent_symbol(agent_idx)


def box_reward_var(box_idx):
    return 'reward_%s' % box_symbol(box_idx)


def push_penalty_var(agent_idx, box_idx):
    return 'push_penalty_%s_%s' % (agent_symbol(agent_idx), box_symbol(box_idx))


def cpush_penalty_var(agent1_idx, agent2_idx, box_idx):
    return 'push_penalty_%s_%s_%s' % (agent_symbol(agent1_idx), agent_symbol(agent2_idx), box_symbol(box_idx))


def action_var():
    return 'main_action_var'


def move_action(direction_symbol, agent_idx):
    return 'action_move_%s_%s' % (direction_symbol, agent_symbol(agent_idx))


def push_action(direction_symbol, box_idx, agent_idx):
    return 'action_push_%s_%s_%s' % (direction_symbol, box_symbol(box_idx), agent_symbol(agent_idx))


def cpush_action(direction_symbol, box_idx, *agents_indices):
    return 'action_cpush_%s_%s_' % (direction_symbol, box_symbol(box_idx)) + \
           '_'.join([agent_symbol(agent_idx) for agent_idx in agents_indices])


def sense_action(box_idx, agent_idx):
    return 'action_sense_%s_%s' % (box_symbol(box_idx), agent_symbol(agent_idx))


def idle_action():
    return 'action_idle'


def get_agent_idx_from_symbol(agent_symbol):
    return int(agent_symbol[len('agent'):]) - 1


def is_public_action(action):
    return is_push_action(action) or is_cpush_action(action)


def is_push_action(action):
    return action.startswith('action_push')


def is_cpush_action(action):
    return action.startswith('action_cpush')


def is_sense_action(action):
    return action.startswith('action_sense')


def is_idle_action(action):
    return action == idle_action()


def is_collaborative_action(action):
    return action.startswith('action_cpush')


def extract_agents_from_action(action):
    """Note that agents are actually the agents, as they are the action deciders"""
    return [comp for comp in action.split('_') if comp.startswith('agent')]


def extract_boxes_from_action(action):
    return [comp for comp in action.split('_') if comp.startswith('box')]


def get_direction(action):
    for d in DIRECTION_SYMBOLS:
        if d in action:
            return d
    return None


def extract_objectives_from_action(action, metadata=None):
    boxes = extract_boxes_from_action(action)
    is_sense = is_sense_action(action)
    if len(boxes) > 0:  # push or sense
        if is_sense:  # sense
            assert len(boxes) == 1
            return [minobs_symbol(boxes[0])]
        else:  # push
            return boxes
    else:  # move or idle
        objectives = extract_agents_from_action(action)
        if metadata is not None:
            for objective in extract_metadata_objectives_from_action(action, metadata):
                objectives.append(objective)
        return objectives


def extract_preconditions_from_action(action):
    boxes = extract_boxes_from_action(action)
    agents = extract_agents_from_action(action)
    if len(boxes) > 0:  # push or sense
        preconditions = [*agents]
        preconditions = [*preconditions, *boxes]
    else:  # move or idle
        preconditions = agents
    return preconditions


def extract_agents_from_obsvar(obsvar):
    return [comp for comp in obsvar.split('_') if comp.startswith('agent')]


def extract_agent_from_obsvar(obsvar):
    agents = extract_agents_from_obsvar(obsvar)
    if len(agents) != 1:
        raise Exception("Expected exactly one agent")
    return agents[0]


def parse_metadata(metadata_xml_elem):
    return {}


def extract_metadata_objectives_from_action(action, metadata):
    return []


def state_measure(state):
    box_idx = 0
    measure = 0
    while True:
        curr_symbol = box_symbol(box_idx)
        if curr_symbol in state:
            measure += (state[curr_symbol] != '0')
        else:  # reached after last box
            return measure
        box_idx += 1


def is_final_state(state):
    return state_measure(state) == 0


def is_empty_state(state):
    return state_measure(state) == 0
