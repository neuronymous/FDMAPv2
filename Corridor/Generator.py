from itertools import combinations
from random import randint
import os

import numpy as np
from jinja2 import Environment, FileSystemLoader

from Corridor import CorridorConstants
from POMDPX import POMDPXConstants
from RS import RockSamplingConstants
from RS.RockSamplingConstants import DIRECTION_SYMBOLS

""" Note, tile stands for (i,j) base-1 index, where i is the row, j is the column
pos index stands for the linear index of the tile, base-0, which depends on the width as well
"""
# Configuration

# Environment Parameters
PLACE_HOLDER_SYMBOL = 'P'


def calculate_pairs(num_elems):
    return [(i, j) for i, j in combinations(range(0, num_elems), 2)]


def calculate_action_list(num_agents):
    all_actions = []
    for agent_idx in range(num_agents):
        all_actions.append(CorridorConstants.move_action(agent_idx))

    for agent_idx_pair in calculate_pairs(num_agents):
        all_actions.append(CorridorConstants.click_action(*agent_idx_pair))

    all_actions.append(CorridorConstants.idle_action())

    return all_actions


def calculate_tile_index(w, tile):
    return (tile[0]) * w + (tile[1])


def calculate_tile_from_idx(w, h, tile_idx):
    return int(tile_idx / w), tile_idx % h


def create_combination_template(length, *args):
    res = [RockSamplingConstants.WILDCARD for _ in range(length)]
    for i in args:
        res[i] = PLACE_HOLDER_SYMBOL
    return ' '.join(res)


def calculate_direction(src_tile, dst_tile):
    x1, y1 = src_tile[0], src_tile[1]
    x2, y2 = dst_tile[0], dst_tile[1]
    if y2 - y1 > 0:
        return RockSamplingConstants.right()
    elif y1 - y2 > 0:
        return RockSamplingConstants.left()
    elif x2 - x1 > 0:
        return RockSamplingConstants.down()
    else:
        return RockSamplingConstants.up()


def cast_to_template(template, cast):
    return template.replace(PLACE_HOLDER_SYMBOL, cast)


def calculate_all_tiles(w, h):
    return [(i, j) for j in range(0, w) for i in range(0, h)]


def calculate_neighbors_dict(w, h):
    res = {}
    tiles = calculate_all_tiles(w, h)
    for tile in tiles:
        cur_neighbors = []
        i, j = tile[0], tile[1]
        if i > 1:
            cur_neighbors.append((i - 1, j))
        if i < h:
            cur_neighbors.append((i + 1, j))
        if j > 1:
            cur_neighbors.append((i, j - 1))
        if j < w:
            cur_neighbors.append((i, j + 1))
        res[tile] = cur_neighbors
    return res


def matrix_to_string(mat):
    return '\n'.join('\t'.join('%0.2f' % x for x in y) for y in mat)


def calculate_move_matrix(corridor_length, succ_prob):
    num_tiles = corridor_length
    res = np.zeros(shape=(num_tiles, num_tiles))  # can use sparse matrix instead
    for src_tile_idx in range(num_tiles):
        dst_tile_idx = min(src_tile_idx + 1, corridor_length - 1)
        if dst_tile_idx == src_tile_idx:
            res[src_tile_idx, dst_tile_idx] = 1.0
        else:
            res[src_tile_idx, src_tile_idx] = 1.0 - succ_prob
            res[src_tile_idx, dst_tile_idx] = succ_prob
    return res


def calculate_euclidian_distance_in_grid(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const):
    all_tiles = [calculate_tile_from_idx(w, h, i) for i in range(w * h)]
    distances = {(i, j): calculate_euclidian_distance_in_grid(all_tiles[i], all_tiles[j]) for i, j in
                 combinations(range(w * h), 2)}

    res = {}
    for rock_idx, rock_tile_idx in enumerate(rock_positions):
        curr_mat = np.zeros(shape=(w * h, 3))
        for curr_pos in range(w * h):
            try:
                distance_to_rock = distances[rock_tile_idx, curr_pos]
            except KeyError:
                try:
                    distance_to_rock = distances[curr_pos, rock_tile_idx]
                except KeyError:
                    distance_to_rock = 0
            succ_prob = 0.5 + 0.5 / (sense_decay_const ** distance_to_rock)
            curr_mat[curr_pos][RockSamplingConstants.good_idx] = succ_prob
            curr_mat[curr_pos][RockSamplingConstants.bad_idx] = 1 - succ_prob
            curr_mat[curr_pos][RockSamplingConstants.null_idx] = 0
        res[RockSamplingConstants.rock_symbol(rock_idx)] = curr_mat
    return res


def calculate_bad_sense_matrices(w, h, rock_positions, sense_decay_const):
    res = calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const)
    good, bad = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx
    for rock, sense_matrix in res.items():
        sense_matrix[:, [good, bad]] = sense_matrix[:, [bad, good]]
    return res


def calculate_sense_martices(w, h, rock_positions, sense_decay_const):
    res = {}
    good_matrices = calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const)
    bad_matrices = calculate_bad_sense_matrices(w, h, rock_positions, sense_decay_const)
    for rock_idx in range(len(rock_positions)):
        rock_symbol = RockSamplingConstants.rock_symbol(rock_idx)
        res[rock_symbol] = {RockSamplingConstants.good_quality(): good_matrices[rock_symbol],
                            RockSamplingConstants.bad_quality(): bad_matrices[rock_symbol]}
    return res


def get_rock_sample_reward_matrix(good_sample_reward, bad_sample_penalty):
    res = np.zeros(shape=(2, 2))
    good, bad = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx

    res[bad][bad] = -bad_sample_penalty
    res[bad][good] = 0  # can't happen
    res[good][bad] = good_sample_reward
    res[good][good] = 0  # no effect
    return res


def generate_random_tile(w, h):
    return randint(a=0, b=h - 1), randint(a=0, b=w - 1)


def project_direction_matrix_to_control_area(direction_matrix, control_area):
    num_rows, num_cols = direction_matrix.shape
    res = direction_matrix.copy()
    for i in range(num_rows):
        if i not in control_area:
            res[i] = np.zeros(num_cols)
            res[i][i] = 1.0
    return res


def project_sense_matrix_to_control_area(sense_matrix, control_area):
    num_rows, num_obs = sense_matrix.shape
    res = sense_matrix.copy()
    good, bad, null = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx, RockSamplingConstants.null_idx

    for i in range(num_rows):
        if i not in control_area:
            res[i][good] = 0.5
            res[i][bad] = 0.5
            res[i][null] = 0
    return res


def calculate_sample_matrix(sample_prob):
    res = np.zeros(shape=(2, 2))
    good, bad = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx

    res[bad][bad] = 1.0
    res[bad][good] = 0.0
    res[good][bad] = sample_prob
    res[good][good] = 1 - sample_prob
    return res


def generate_template(template_file_name, parameters):
    """
    :param template_file_name: file should be in cwd/templates/template_file_name, in jinja2 format
    :param parameters: dictionary containing parameters for rendering
    :return: the rendered template in string format
    """
    env = Environment(loader=FileSystemLoader('templates'), lstrip_blocks=True, trim_blocks=True,
                      extensions=['jinja2.ext.do'])
    env.globals.update(calculate_action_list=calculate_action_list,
                       calculate_tile_index=calculate_tile_index,
                       create_combination_template=create_combination_template,
                       calculate_pairs=calculate_pairs,
                       cast_to_template=cast_to_template,
                       calculate_move_matrix=calculate_move_matrix,
                       corridor_constants=CorridorConstants,
                       pomdpx_constants=POMDPXConstants,
                       time_zero=POMDPXConstants.time_zero,
                       time_one=POMDPXConstants.time_one,
                       matrix_to_string=matrix_to_string)
    template = env.get_template(template_file_name)
    return template.render(parameters)


def generate_domain_config(corridor_length, num_agents,
                           good_click_reward,
                           bad_click_penalty,
                           move_cost,
                           click_cost,
                           move_prob,
                           agents_positions,
                           bias_constant,
                           discount=0.95,
                           domain_name_suffix=""):
    agents_positions = agents_positions
    agents_positions = [{calculate_tile_index(corridor_length, tile): prob for tile, prob in dist.items()} for dist in
                        agents_positions] if agents_positions is not None else [
        {calculate_tile_index(corridor_length, (0, 0)): 1.0} for _ in
        range(num_agents)]
    res = {"DOMAIN_NAME": "COR-%d_%dA%s" % (
        corridor_length, num_agents, domain_name_suffix),
           "NUM_AGENTS": num_agents,
           "CORRIDOR_LENGTH": corridor_length,
           "DISCOUNT": discount,

           # None yields a uniform distribution
           # Positions are indices not tiles!
           "AGENTS_POSITIONS": agents_positions,

           "PROB_MOVE": move_prob,

           "MOVE_COST": -move_cost,
           "CLICK_COST": -click_cost,

           "GOOD_CLICK_REWARD": good_click_reward,
           "BAD_CLICK_PENALTY": -bad_click_penalty,

           "AGENTS_BIAS": [bias_constant * i for i in range(num_agents)],

           "AGENTS_SYMBOLS": [CorridorConstants.agent_symbol(i) for i in range(num_agents)],
           }
    return res


corridor_length, num_agents = 4, 2
agents_positions = [{(0, 0): 1.0} for _ in range(num_agents)]
good_click_reward = 100
bad_click_penalty = 500
move_cost = 1
click_cost = 0.1
move_prob = 0.5
bias_constant = 0.1

templates = ["cor-domain-template.pomdpx.j2"]
domains = {
    "initial": {"config": generate_domain_config(corridor_length=corridor_length, num_agents=num_agents,
                                                 good_click_reward=good_click_reward,
                                                 bad_click_penalty=bad_click_penalty,
                                                 move_cost=move_cost,
                                                 move_prob=move_prob,
                                                 click_cost=click_cost,
                                                 agents_positions=agents_positions,
                                                 bias_constant=bias_constant,
                                                 domain_name_suffix=""),
                "template": "cor-domain-noobs-template.pomdpx.j2"},
}

problems_path = "../Resources/problems"

for d in domains:
    params = domains[d]["config"]
    domain_name = params["DOMAIN_NAME"]
    output_filename = "%s.pomdpx" % domain_name
    output_path = os.path.join(problems_path, output_filename)
    res = generate_template(template_file_name=domains[d]["template"], parameters=params)
    with open(output_path, "w") as fh:
        fh.write(res)
