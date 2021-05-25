from collections import Counter
from itertools import combinations
from random import randint
import os

import numpy as np
from jinja2 import Environment, FileSystemLoader

from DecTiger import Constants
from DecTiger.Constants import DIRECTION_SYMBOLS, left, right, up, down
from POMDPX import POMDPXConstants

""" Note, tile stands for (i,j) base-1 index, where i is the row, j is the column
pos index stands for the linear index of the tile, base-0, which depends on the width as well
"""
# Configuration

# Environment Parameters
PLACE_HOLDER_SYMBOL = 'P'


def calc_open_effect_matrix(w, h, num_agents, agents_actions):
    """P(tiger_1 | tiger_0 a1_0, a2_0...,  action1, action2...)"""
    num_tiles = w * h
    res = np.ndarray(shape=[num_tiles, num_tiles, *[num_tiles for _ in range(num_agents)],
                            *[len(agents_actions[i]) for i in range(num_agents)]])
    for index in np.ndindex(res.shape):
        tiger_1 = index[0]
        tiger_0 = index[1]
        chosen_agents_locs = index[2:2 + num_agents]
        chosen_actions = [agents_actions[agent_idx][i] for agent_idx, i in enumerate(index[-num_agents:])]
        res[index] = open_effect_prob(tiger_1, tiger_0, chosen_agents_locs, chosen_actions)
    np.set_printoptions(threshold=np.inf)
    # if flatten:
    #     return np.array2string(res.flatten('F'), formatter={'float_kind': lambda x: "%.2f" % x}, separator=' ')[1:-1]
    # else:
    #     return matrix_to_string(res)
    return ' '.join(list(map(str, res.flatten('F'))))


def open_effect_prob(tiger_1, tiger_0, chosen_agents_locs, chosen_actions):
    if any([Constants.is_open_action(action) for action in chosen_actions]):
        return 0.5
    elif all([Constants.is_copen_action(action) for action in chosen_actions]) and all(
            [loc == chosen_agents_locs[0] for loc in chosen_agents_locs]):
        return 0.5
    return 1.0 if tiger_0 == tiger_1 else 0.0


def calc_open_reward_matrix(w, h, num_agents, agents_actions, open_reward, open_penalty, copen_reward,
                            copen_penalty):
    """R(tiger_0, a1_0, a2_0...,  action1, action2...)"""
    num_tiles = w * h
    res = np.ndarray(shape=[num_tiles, *[num_tiles for _ in range(num_agents)],
                            *[len(agents_actions[i]) for i in range(num_agents)]])
    for index in np.ndindex(res.shape):
        tiger = index[0]
        chosen_agents_locs = index[1:1 + num_agents]
        chosen_actions = [agents_actions[agent_idx][i] for agent_idx, i in enumerate(index[-num_agents:])]
        res[index] = open_reward_value(tiger, chosen_agents_locs, chosen_actions, open_reward, open_penalty,
                                       copen_reward,
                                       copen_penalty)
    np.set_printoptions(threshold=np.inf)
    # if flatten:
    #     return np.array2string(res.flatten('F'), formatter={'float_kind': lambda x: "%.2f" % x}, separator=' ')[1:-1]
    # else:
    #     return matrix_to_string(res)
    return ' '.join(list(map(str, res.flatten('F'))))


def open_reward_value(tiger, chosen_agents_locs, chosen_actions, open_reward, open_penalty, copen_reward,
                      copen_penalty):
    agents_pushing_alone = [Constants.is_open_action(action) for action in chosen_actions]
    agents_pushing_collab = [Constants.is_copen_action(action) for action in chosen_actions]

    if sum(agents_pushing_collab) == len(agents_pushing_collab):
        if all([loc == chosen_agents_locs[0] for loc in chosen_agents_locs]):
            if chosen_agents_locs[0] == tiger:
                return copen_penalty * sum(agents_pushing_collab)
            else:
                return copen_reward * sum(agents_pushing_collab)
        return 0.0

    else:
        total_reward = 0.0
        for agent_idx, is_agent_pushes_alone in enumerate(agents_pushing_alone):
            if is_agent_pushes_alone:
                if chosen_agents_locs[agent_idx] == tiger:
                    total_reward += open_penalty
                else:
                    total_reward += open_reward
        return total_reward


def calculate_action_list(num_agents):
    all_actions = []
    all_actions.append(Constants.idle_action())

    for a_i in range(num_agents):
        for d in DIRECTION_SYMBOLS:
            all_actions.append(Constants.move_action(d, a_i))
        all_actions.append(Constants.listen_action(a_i))
        all_actions.append(Constants.open_action(a_i))

    if num_agents >= 2:  # No joint actions if we're talking about single agent
        agents_pairs_indices = calculate_pairs(num_agents)
        for a1_i, a2_i in agents_pairs_indices:
            all_actions.append(Constants.copen_action(a1_i, a2_i))

    return list(set(all_actions))


def actions_per_agent(num_agents):
    res = []
    for a_i in range(num_agents):
        curr_agent_actions = []
        curr_agent_actions.append(Constants.idle_action())

        for d in DIRECTION_SYMBOLS:
            curr_agent_actions.append(Constants.move_action(d, a_i))
        curr_agent_actions.append(Constants.listen_action(a_i))
        curr_agent_actions.append(Constants.open_action(a_i))

        if num_agents >= 2:  # No joint actions if we're talking about single agent
            agents_pairs_indices = calculate_pairs(num_agents)
            for a1_i, a2_i in agents_pairs_indices:
                if a1_i == a_i or a2_i == a_i:
                    curr_agent_actions.append(Constants.copen_action(a1_i, a2_i))
        res.append(curr_agent_actions)
    return res


def calculate_tile_index(w, tile):
    return (tile[0] - 1) * w + (tile[1] - 1)


def create_combination_template(length, *args):
    res = ['*' for i in range(length)]
    for i in args:
        res[i] = PLACE_HOLDER_SYMBOL
    return ' '.join(res)


def calculate_direction(src_tile, dst_tile):
    x1, y1 = src_tile[0], src_tile[1]
    x2, y2 = dst_tile[0], dst_tile[1]
    if y2 - y1 > 0:
        return right()
    elif y1 - y2 > 0:
        return left()
    elif x2 - x1 > 0:
        return down()
    else:
        return up()


def cast_to_template(template, cast):
    return template.replace(PLACE_HOLDER_SYMBOL, cast)


def calculate_all_tiles(w, h):
    return [(i, j) for i in range(1, h + 1) for j in range(1, w + 1)]


def calculate_pairs(num_elems):
    return [(i, j) for i, j in combinations(range(0, num_elems), 2)]


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


def calculate_direction_matrices(w, h, succ_prob, to_string=True):
    def calculate_dest(src, direction):
        i, j = src[0], src[1]
        if direction == left():
            j -= 1
        elif direction == right():
            j += 1
        elif direction == up():
            i -= 1
        else:
            i += 1

        return (i, j) if 1 <= i <= h and 1 <= j <= w else None

    res = {}
    tiles = calculate_all_tiles(w, h)
    num_tiles = w * h
    for d in DIRECTION_SYMBOLS:
        cur_matrix = np.zeros(shape=(num_tiles, num_tiles))  # can use sparse matrix instead
        movement_map = {src_tile: calculate_dest(src_tile, d) for src_tile in tiles}
        for src_tile in tiles:
            dst_tile = movement_map[src_tile]
            src_idx = calculate_tile_index(w, src_tile)
            if dst_tile is None:
                cur_matrix[src_idx, src_idx] = 1.0
            else:
                dst_idx = calculate_tile_index(w, dst_tile)
                cur_matrix[src_idx, src_idx] = 1.0 - succ_prob
                cur_matrix[src_idx, dst_idx] = succ_prob
        res[d] = '\n'.join('\t'.join('%0.2f' % x for x in y) for y in cur_matrix) if to_string else cur_matrix
    return res


def matrix_to_string(mat):
    return '\n'.join('\t'.join('%0.2f' % x for x in y) for y in mat)


def get_identity_matrix(n):
    return matrix_to_string(np.eye(n))


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
                       calculate_direction=calculate_direction,
                       cast_to_template=cast_to_template,
                       calculate_all_tiles=calculate_all_tiles,
                       calculate_pairs=calculate_pairs,
                       calculate_neighbors_dict=calculate_neighbors_dict,
                       calculate_direction_matrices=calculate_direction_matrices,
                       get_identity_matrix=get_identity_matrix,
                       time_zero=POMDPXConstants.time_zero,
                       time_one=POMDPXConstants.time_one,
                       constants=Constants,
                       pomdpxconstants=POMDPXConstants,
                       actions_per_agent=actions_per_agent,
                       calc_open_reward_matrix=calc_open_reward_matrix,
                       calc_open_effect_matrix=calc_open_effect_matrix
                       )
    template = env.get_template(template_file_name)
    return template.render(parameters)


def generate_random_tile(w, h):
    return randint(a=1, b=h), randint(a=1, b=w)


class InstanceConfig:
    default_discount = 0.99
    default_bias_constant = 1.05
    default_domain_suffix = ""
    default_width = 2
    default_num_agents = 2
    default_move_prob = 1.0
    default_listen_prob = 0.85

    @staticmethod
    def generate_config_dict(move_cost, listen_cost, open_cost, copen_cost,
                             open_penalty, copen_penalty,
                             open_reward, copen_reward,
                             is_pgmx,
                             move_prob=default_move_prob, listen_prob=default_listen_prob,
                             bias_constant=default_bias_constant,
                             width=default_width,
                             num_agents=default_num_agents,
                             discount=default_discount,
                             agent_pos_dist=None,
                             domain_name_suffix='',
                             is_dec=False):
        problem_name = Constants.PROBLEM_NAME
        if is_pgmx:
            template = "dt-domain-template.pgmx.j2"
        else:
            template = "dt-domain-template-team.pomdpx.j2"  # if not is_dec else "dt-domain-template-dec.pomdpx.j2"
        if domain_name_suffix != "":
            domain_name_suffix = "_%s" % domain_name_suffix
        res = {
            "config": {"DOMAIN_NAME": "%s-%dx1_%dA%s" % (problem_name, width, num_agents, domain_name_suffix),
                       "NUM_AGENTS": num_agents,
                       "WIDTH": width,
                       "MOVE_COST": -move_cost, "LISTEN_COST": -listen_cost, "OPEN_COST": -open_cost,
                       "COPEN_COST": -copen_cost,
                       "OPEN_PENALTY": -open_penalty, "COPEN_PENALTY": -copen_penalty,
                       "OPEN_REWARD": open_reward, "COPEN_REWARD": copen_reward,
                       "MOVE_PROB": move_prob, "LISTEN_PROB": listen_prob,
                       "AGENTS_POS_DIST": [{generate_random_tile(width, 1): 1.0} for _ in
                                           range(num_agents)] if agent_pos_dist is None else agent_pos_dist,
                       "AGENTS_BIAS": [bias_constant ** i for i in range(num_agents)],
                       "AGENTS": [Constants.agent_symbol(a_i) for a_i in range(num_agents)],
                       "TIGER": Constants.tiger_symbol(),
                       "DISCOUNT": discount,
                       "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS
                       },

            "template": template
        }
        return res


width = 2
num_agents = 2
move_cost, listen_cost, open_cost, copen_cost = 0.01, 1, 0.01, 0.01
open_penalty, copen_penalty = 100, 25
open_reward, copen_reward = 10, 10
agent_pos_dist = [{(1, 1): 1.0}, {(1, 2): 1.0}]
name_suffix = ''
bias_constant = 1

domains = {
    "1": {**InstanceConfig.generate_config_dict(
        move_cost, listen_cost, open_cost, copen_cost,
        open_penalty, copen_penalty,
        open_reward, copen_reward, bias_constant=bias_constant,
        is_pgmx=False)},
    # "2": {**InstanceConfig.generate_config_dict(
    #     move_cost, listen_cost, open_cost, copen_cost,
    #     open_penalty, copen_penalty,
    #     open_reward, copen_reward, domain_name_suffix="DEC" + name_suffix,
    #     is_pgmx=False, is_dec=True)},
    "3": {**InstanceConfig.generate_config_dict(
        move_cost, listen_cost, open_cost, copen_cost,
        open_penalty, copen_penalty,
        open_reward, copen_reward, bias_constant=bias_constant,
        is_pgmx=True)}
}

problems_path = "../Resources/problems"

for d in domains:
    params = domains[d]["config"]
    domain_name = params["DOMAIN_NAME"]
    output_filename = "%s.%s" % (domain_name, domains[d]["template"].split('.')[-2])
    output_path = os.path.join(problems_path, output_filename)
    res = generate_template(template_file_name=domains[d]["template"], parameters=params)
    with open(output_path, "w") as fh:
        fh.write(res)
