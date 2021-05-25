from typing import Callable

DIRECTION_SYMBOLS = ['left', 'right', 'up', 'down']
ROCK_QUALITY_SYMBOLS = ['good', 'bad']
NULL_OBS = 'null'
ROCK_OBSERVATION_SYMBOLS = [*ROCK_QUALITY_SYMBOLS, NULL_OBS]


# Metadata:
# RocksLocations: List(int)
# ControlAreas: List(ControlArea: List(int))

def good_obs():
    return ROCK_QUALITY_SYMBOLS[0]


def bad_obs():
    return ROCK_QUALITY_SYMBOLS[1]


left: Callable[[], str] = lambda: DIRECTION_SYMBOLS[0]
right: Callable[[], str] = lambda: DIRECTION_SYMBOLS[1]
up: Callable[[], str] = lambda: DIRECTION_SYMBOLS[2]
down: Callable[[], str] = lambda: DIRECTION_SYMBOLS[3]
WILDCARD = '*'
good_idx = 0
bad_idx = 1
null_idx = 2
DECLARE_METADATA_ROCKSLOCATIONS = 'RocksPositions'
DECLARE_METADATA_CONTROLAREAS = 'ControlAreas'
DECLARE_METADATA_CONTROLAREA = 'ControlArea'


def good_quality():
    return ROCK_QUALITY_SYMBOLS[0]


def bad_quality():
    return ROCK_QUALITY_SYMBOLS[1]


def agent_symbol(agent_idx):
    return car_symbol(agent_idx)


def car_symbol(car_idx):
    return 'car%d' % (car_idx + 1)


def rock_symbol(rock_idx):
    return 'rock%d' % (rock_idx + 1)


def obs_symbol(rock_idx, car_idx):
    return 'obs_%s_%s' % (rock_symbol(rock_idx), car_symbol(car_idx))

def minobs_symbol(car_idx):
    return 'obs_%s' % (car_symbol(car_idx))




def car_cost_var(car_idx):
    return 'cost_%s' % car_symbol(car_idx)


def rock_penalties_var(rock_idx, car_idx):
    return 'penalty_%s_%s' % (rock_symbol(rock_idx), car_symbol(car_idx))


def rock_rewards_var(car_idx):
    return 'reward_rocks_%s' % car_symbol(car_idx)


def action_var():
    return 'main_action_var'


def move_action(direction_symbol, car_idx):
    return 'action_move_%s_%s' % (direction_symbol, car_symbol(car_idx))


def private_sample_action(car_idx):
    return 'action_psample_%s' % car_symbol(car_idx)


def shared_sample_action(car_idx):
    return 'action_ssample_%s' % car_symbol(car_idx)


def sense_action(rock_idx, car_idx):
    return 'action_sense_%s_%s' % (rock_symbol(rock_idx), car_symbol(car_idx))


def idle_action():
    return 'action_idle'


def get_car_from_action(action):
    if 'car' in action:
        return action.split('_')[-1]
    return None


def get_agent_from_action(action):
    return get_car_from_action(action)


def get_rock_symbol_from_action(action):
    if is_sense_action(action):
        return action.split('_')[-2]
    return None


def get_agent_idx_from_symbol(agent_symbol):
    return int(agent_symbol[-1]) - 1


def is_public_action(action):
    return is_shared_sample_action(action)


def is_private_sample_action(action):
    return action.startswith('action_psample')


def is_shared_sample_action(action):
    return action.startswith('action_ssample')


def is_sense_action(action):
    return action.startswith('action_sense')


def is_idle_action(action):
    return action == idle_action()


def is_collaborative_action(action):
    return False


def extract_agents_from_action(action):
    """Note that agents are actually the controllers, as they are the action deciders"""
    return [comp for comp in action.split('_') if comp.startswith('car')]


def extract_agents_from_obsvar(obsvar):
    return [comp for comp in obsvar.split('_') if comp.startswith('car')]


def extract_agent_from_obsvar(obsvar):
    agents = extract_agents_from_obsvar(obsvar)
    if len(agents) != 1:
        raise Exception("Expected exactly one agent")
    return agents[0]


def parse_metadata(metadata_xml_elem):
    shared_rocks_positions = metadata_xml_elem.find('SharedRocksPositions').text.strip(' ').split(' ')
    private_rocks_positions = metadata_xml_elem.find('PrivateRocksPositions').text.strip(' ').split(' ')
    rocks_positions = private_rocks_positions + shared_rocks_positions
    return {'rocks_positions': rocks_positions,
            'control_areas': [control_area.text.strip(' ').split(' ') for control_area in
                              metadata_xml_elem.find('ControlAreas').findall('ControlArea')]}


def get_car_rock_symbols(car_index, metadata):
    control_area = metadata['control_areas'][car_index]
    return [rock_symbol(i) for i, pos in enumerate(metadata['rocks_positions']) if pos in control_area]


def is_move_action(action):
    return 'move' in action


def extract_preconditions_from_action(action, metadata):
    try:
        car = extract_agents_from_action(action)[0]
    except:
        return []
    car_index = get_agent_idx_from_symbol(car)
    if is_private_sample_action(action) or is_shared_sample_action(action):
        rocks_symbols = get_car_rock_symbols(car_index, metadata=metadata)
        preconditions = [car, *rocks_symbols]
    elif is_sense_action(action):
        rock = get_rock_symbol_from_action(action)
        preconditions = [car, rock]
    elif is_move_action(action):  # move
        preconditions = [car]
    else:  # idle
        preconditions = []
    return preconditions


def extract_objectives_from_action(action, metadata=None):
    try:
        car = extract_agents_from_action(action)[0]
    except:  # idle
        return []
    car_index = get_agent_idx_from_symbol(car)
    if is_private_sample_action(action) or is_shared_sample_action(action):
        rocks_symbols = get_car_rock_symbols(car_index, metadata=metadata)
        objectives = rocks_symbols
    elif is_sense_action(action):
        objectives = []
    elif is_move_action(action):  # move
        objectives = [car]
    else:  # idle
        objectives = []
    return objectives


def state_measure(state):
    rock_idx = 0
    measure = 0
    while True:
        rock = rock_symbol(rock_idx)
        if rock in state:
            measure += (state[rock] == good_quality())
        else:  # reached after last box
            return measure
        rock_idx += 1


def is_final_state(state):
    return state_measure(state) == 0


def is_empty_state(state):
    return state_measure(state) == 0
