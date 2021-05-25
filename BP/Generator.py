from collections import Counter
from itertools import combinations
from random import randint
import os

import numpy as np
from jinja2 import Environment, FileSystemLoader

from BP import Constants
from BP.Constants import DIRECTION_SYMBOLS, left, right, up, down
from POMDPX import POMDPXConstants

""" Note, tile stands for (i,j) base-1 index, where i is the row, j is the column
pos index stands for the linear index of the tile, base-0, which depends on the width as well
"""
# Configuration

# Environment Parameters
PLACE_HOLDER_SYMBOL = 'P'


def calculate_action_list(num_agents, num_boxes, heavy_boxes_indices):
    all_actions = []

    all_actions.append(Constants.idle_action())

    for a_i in range(num_agents):
        for d in DIRECTION_SYMBOLS:
            all_actions.append(Constants.move_action(d, a_i))
            for b_i in range(num_boxes):
                all_actions.append(Constants.sense_action(b_i, a_i))
                if b_i not in heavy_boxes_indices:
                    all_actions.append(Constants.push_action(d, b_i, a_i))

    if num_agents >= 2:  # No joint actions if we're talking about single agent
        agents_pairs_indices = calculate_pairs(num_agents)
        for b_i in heavy_boxes_indices:
            for a1_i, a2_i in agents_pairs_indices:
                for d in DIRECTION_SYMBOLS:
                    all_actions.append(Constants.cpush_action(d, b_i, a1_i, a2_i))

    return list(set(all_actions))


def actions_per_agent(num_agents, num_boxes, heavy_boxes_indices):
    res = []
    for a_i in range(num_agents):
        curr_agent_actions = []
        curr_agent_actions.append(Constants.idle_action())

        for b_i in range(num_boxes):
            curr_agent_actions.append(Constants.sense_action(b_i, a_i))
            if b_i not in heavy_boxes_indices:
                for d in DIRECTION_SYMBOLS:
                    curr_agent_actions.append(Constants.push_action(d, b_i, a_i))

        for d in DIRECTION_SYMBOLS:
            curr_agent_actions.append(Constants.move_action(d, a_i))

        if num_agents >= 2:  # No joint actions if we're talking about single agent
            agents_pairs_indices = calculate_pairs(num_agents)
            for b_i in heavy_boxes_indices:
                for a1_i, a2_i in agents_pairs_indices:
                    if a_i == a1_i or a_i == a2_i:
                        for d in DIRECTION_SYMBOLS:
                            curr_agent_actions.append(Constants.cpush_action(d, b_i, a1_i, a2_i))
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


def calculate_transitions_vec(src_idx, dst_idx, success_prob, num_tiles):
    res = np.zeros(num_tiles)
    res[src_idx] = round(1 - success_prob, 2)
    res[dst_idx] = round(success_prob, 2)
    return [str(i) for i in res]


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


def calculate_push_matrix(w, h, direction, succ_prob, agent_tile):
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

    tiles = calculate_all_tiles(w, h)
    num_tiles = w * h
    res = np.eye(num_tiles)
    dst_idx = calculate_tile_index(w, calculate_dest(agent_tile, direction))
    src_idx = calculate_tile_index(w, agent_tile)
    res[src_idx][src_idx] = 1 - succ_prob
    res[src_idx][dst_idx] = succ_prob
    return matrix_to_string(res)


def calculate_target_indices(target_tiles, w):
    return [calculate_tile_index(w, tile) for tile in target_tiles]


def matrix_to_string(mat):
    return '\n'.join('\t'.join('%0.2f' % x for x in y) for y in mat)


def calculate_box_reward_matrix(w, h, target_indices, reward, penalty):
    num_tiles = w * h
    res = np.zeros(shape=(num_tiles, num_tiles))
    for src in range(num_tiles):
        for dst in range(num_tiles):
            if src in target_indices and dst not in target_indices:
                res[src][dst] = reward
            elif src not in target_indices and dst in target_indices:
                res[src][dst] = penalty
            else:
                res[src][dst] = 0
    return matrix_to_string(res)


def get_identity_matrix(n):
    return matrix_to_string(np.eye(n))


def calc_light_box_push_matrix(w, h, num_agents, agents_actions, push_success_prob, flatten=False):
    """P(b_1|b_0, a1, a2, ..., an, action1, action2, ..., action_n)"""
    num_tiles = w * h
    res = np.ndarray(shape=[num_tiles, num_tiles, *[num_tiles for _ in range(num_agents)],
                            *[len(agents_actions[i]) for i in range(num_agents)]])

    direction_matrices = calculate_direction_matrices(w, h, push_success_prob, to_string=False)

    for index in np.ndindex(res.shape):
        b1 = index[0]
        b0 = index[1]
        chosen_agents_locs = index[2:2 + num_agents]
        chosen_actions = [agents_actions[agent_idx][i] for agent_idx, i in enumerate(index[-num_agents:])]
        res[index] = light_box_cond_prob(b1, b0, chosen_agents_locs, chosen_actions,
                                         direction_matrices)
    np.set_printoptions(threshold=np.inf)
    # if flatten:
    #     return np.array2string(res.flatten('F'), formatter={'float_kind': lambda x: "%.2f" % x}, separator=' ')[1:-1]
    # else:
    #     return matrix_to_string(res)
    return ' '.join(list(map(str, res.flatten('F'))))


def light_box_cond_prob(b1, b0, chosen_agents_locs, chosen_actions, direction_matrices):
    loc_matching_agents = [b0 == agent_loc for agent_loc in chosen_agents_locs]
    good_pushing_agents = [is_same_loc_agent and Constants.is_push_action(action) for
                           is_same_loc_agent, action in
                           zip(loc_matching_agents, chosen_actions)]
    if sum(good_pushing_agents) != 1:
        if sum(good_pushing_agents) == 0:
            return 0.0 if b0 != b1 else 1.0
        # if there's more than one good pushing agent, take the first
    pushing_agent_index = good_pushing_agents.index(True)
    push_direction = Constants.get_direction(chosen_actions[pushing_agent_index])
    return round(direction_matrices[push_direction][b0, b1], 2)


def calc_heavy_box_push_matrix(w, h, num_agents, agents_actions, cpush_success_prob):
    """P(b_1|b_0, a1, a2, ..., an, action1, action2, ..., action_n)"""
    num_tiles = w * h
    res = np.ndarray(shape=[num_tiles, num_tiles, *[num_tiles for _ in range(num_agents)],
                            *[len(agents_actions[i]) for i in range(num_agents)]])

    direction_matrices = calculate_direction_matrices(w, h, cpush_success_prob, to_string=False)

    for index in np.ndindex(res.shape):
        b1 = index[0]
        b0 = index[1]
        chosen_agents_locs = index[2:2 + num_agents]
        chosen_actions = [agents_actions[agent_idx][i] for agent_idx, i in enumerate(index[-num_agents:])]
        res[index] = heavy_box_cond_prob(b1, b0, chosen_agents_locs, chosen_actions,
                                         direction_matrices)
    np.set_printoptions(threshold=np.inf)
    return np.array2string(res.flatten('F'), formatter={'float_kind': lambda x: "%.2f" % x}, separator=' ')[1:-1]


def heavy_box_cond_prob(b1, b0, chosen_agents_locs, chosen_actions, direction_matrices):
    loc_matching_agents = [b0 == agent_loc for agent_loc in chosen_agents_locs]
    good_pushing_agents = [is_same_loc_agent and Constants.is_cpush_action(action) for
                           is_same_loc_agent, action in
                           zip(loc_matching_agents, chosen_actions)]
    if sum(good_pushing_agents) < 2:
        return 0.0 if b0 != b1 else 1.0
    else:
        cpush_directions = Counter([Constants.get_direction(chosen_actions[i]) for i in range(num_agents) if
                                    good_pushing_agents[i] == True])
        if max(cpush_directions.values()) < 2:
            return 0.0 if b0 != b1 else 1.0

        # If there are more than two agents collaborating, take the first collaborating group
        push_direction = list(filter(lambda x: cpush_directions[x] >= 2, cpush_directions))[0]
        return round(direction_matrices[push_direction][b0, b1], 2)


def calculate_push_penalty_matrix(w, h, penalty, flatten=True, flip=False):
    num_tiles = w * h
    if not flip:
        res = np.ones(shape=(num_tiles, num_tiles)) * (-penalty) + np.eye(num_tiles) * penalty
    else:
        res = np.eye(num_tiles) * -penalty
    np.set_printoptions(threshold=np.inf)
    if flatten:
        return np.array2string(res.flatten('F'), formatter={'float_kind': lambda x: "%.2f" % x}, separator=' ')[1:-1]
    else:
        return matrix_to_string(res)


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
                       calculate_transitions_vec=calculate_transitions_vec,
                       calculate_all_tiles=calculate_all_tiles,
                       calculate_pairs=calculate_pairs,
                       calculate_neighbors_dict=calculate_neighbors_dict,
                       calculate_direction_matrices=calculate_direction_matrices,
                       calculate_target_indices=calculate_target_indices,
                       calculate_box_reward_matrix=calculate_box_reward_matrix,
                       get_identity_matrix=get_identity_matrix,
                       calculate_push_matrix=calculate_push_matrix,
                       time_zero=POMDPXConstants.time_zero,
                       time_one=POMDPXConstants.time_one,
                       bpconstants=Constants,
                       pomdpxconstants=POMDPXConstants,
                       actions_per_agent=actions_per_agent,
                       calc_light_box_push_matrix=calc_light_box_push_matrix,
                       calc_heavy_box_push_matrix=calc_heavy_box_push_matrix,
                       calculate_push_penalty_matrix=calculate_push_penalty_matrix
                       )
    template = env.get_template(template_file_name)
    return template.render(parameters)


def generate_random_tile(w, h):
    return randint(a=1, b=h), randint(a=1, b=w)


def generate_domain_config(width, height, num_agents, num_boxes, heavy_boxes_indices, discount=0.99,
                           prob_move=1.0, prob_push=0.8, prob_cpush=0.8, prob_sense=1.0, move_cost=3, push_cost=5,
                           sense_cost=1.0, cpush_cost=5.0,
                           goal_reward=300, ungoal_penalty=10000, domain_name_suffix='',
                           agent_pos_dist=None, boxes_pos_dist=None, bias_constant=0.0, move_reward=5,
                           unmove_penalty=6, is_pgmx=False, is_pen=False):
    problem_name = "BP" if not is_pen else "BPPEN"
    res = {"DOMAIN_NAME": "%s-%dx%d_%dA_%dH_%dL%s" % (
        problem_name, width, height, num_agents, len(heavy_boxes_indices), num_boxes - len(heavy_boxes_indices),
        domain_name_suffix),
           "NUM_BOXES": num_boxes,
           "NUM_AGENTS": num_agents,
           "WIDTH": width,
           "HEIGHT": height,
           "DISCOUNT": discount,
           "TARGET_TILES": [[(1, 1)]] if not is_pgmx else [(1, 1)],  # can specify one per box
           "HEAVY_BOXES": heavy_boxes_indices,
           # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
           # None yields a uniform distribution
           "AGENTS_POS_DIST": [{generate_random_tile(width, height): 1.0} for _ in
                               range(num_agents)] if agent_pos_dist is None else agent_pos_dist,
           "BOXES_POS_DIST": [{(1, 1): 0.5, (height, width): 0.5} for _ in
                              range(num_boxes)] if boxes_pos_dist is None else boxes_pos_dist,
           "PROB_MOVE": prob_move,
           "PROB_PUSH": prob_push,
           "PROB_CPUSH": prob_cpush,

           "MOVE_COST": -move_cost,
           "PUSH_COST": -push_cost,
           "CPUSH_COST": -cpush_cost,
           "SENSE_COST": -sense_cost,

           "GOAL_REWARD": goal_reward,
           "UNGOAL_PENALTY": -ungoal_penalty,

           "MOVE_REWARD": move_reward,
           "UNMOVE_PENALTY": -unmove_penalty,

           "PUSH_PENALTY": push_penalty,

           "PROB_OBS_BOX": prob_sense,

           "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS,

           "AGENTS": [Constants.agent_symbol(a_i) for a_i in range(num_agents)],
           "BOXES": [Constants.box_symbol(b_i) for b_i in range(num_boxes)],

           "AGENTS_BIAS": [bias_constant ** i for i in range(num_agents)]}
    return res


width, height = 2, 1
num_agents, num_boxes = 2, 1
heavy_boxes_indices = [0]
goal_reward, ungoal_penalty = 500 if len(heavy_boxes_indices) == 0 else 1000, 10000
push_penalty = 101  # > gamma*(push_reward / collaborating_agents) * fail_prob
move_cost, sense_cost, push_cost, cpush_cost = 10, 1, 30, 20
agent_pos_dist = [{(1, 2): 1.0}, {(2, 1): 1.0}]  # , {(2, 3): 1.0}]
boxes_pos_dist = None
assert len(agent_pos_dist) == num_agents
bias_constant = 1.05  # may be worth investigating
prob_move, prob_cpush, prob_push = 1, 0.8, 0.8
name_suffix = ''
PUSH_PEN = False
domains = {
    "team-pomdpx": {
        "config": generate_domain_config(width=width, height=height, num_agents=num_agents, num_boxes=num_boxes,
                                         heavy_boxes_indices=heavy_boxes_indices, goal_reward=goal_reward,
                                         ungoal_penalty=ungoal_penalty,
                                         move_cost=move_cost, sense_cost=sense_cost, push_cost=push_cost,
                                         cpush_cost=cpush_cost,
                                         agent_pos_dist=agent_pos_dist, boxes_pos_dist=boxes_pos_dist,
                                         bias_constant=bias_constant, prob_move=prob_move, prob_cpush=prob_cpush,
                                         prob_push=prob_push,
                                         domain_name_suffix="_TEAM" + name_suffix, is_pgmx=False, is_pen=PUSH_PEN),
        "template": "bp-domain-template-team.pomdpx.j2" if not PUSH_PEN else "bp-domain-template-team-badpen.pomdpx.j2"},
    "dec-pomdpx": {
        "config": generate_domain_config(width=width, height=height, num_agents=num_agents, num_boxes=num_boxes,
                                         heavy_boxes_indices=heavy_boxes_indices, goal_reward=goal_reward,
                                         ungoal_penalty=ungoal_penalty,
                                         move_cost=move_cost, sense_cost=sense_cost, push_cost=push_cost,
                                         cpush_cost=cpush_cost,
                                         agent_pos_dist=agent_pos_dist, boxes_pos_dist=boxes_pos_dist,
                                         bias_constant=bias_constant, prob_move=prob_move, prob_cpush=prob_cpush,
                                         prob_push=prob_push,
                                         domain_name_suffix="_DEC" + name_suffix, is_pgmx=False, is_pen=PUSH_PEN),
        "template": "bp-domain-template-dec.pomdpx.j2" if not PUSH_PEN else "bp-domain-template-dec-badpen.pomdpx.j2"},
    # "pgmx": {"config": generate_domain_config(width=width, height=height, num_agents=num_agents, num_boxes=num_boxes,
    #                                        heavy_boxes_indices=heavy_boxes_indices, goal_reward=goal_reward,
    #                                        ungoal_penalty=ungoal_penalty,
    #                                        move_cost=move_cost, sense_cost=sense_cost, push_cost=push_cost,
    #                                        cpush_cost=cpush_cost,
    #                                        agent_pos_dist=agent_pos_dist, boxes_pos_dist=boxes_pos_dist,
    #                                        bias_constant=bias_constant, prob_move=prob_move, prob_cpush=prob_cpush,
    #                                        prob_push=prob_push,
    #                                        domain_name_suffix="_DEC" + name_suffix, is_pgmx=True, is_pen=PUSH_PEN),
    #       "template": "bp-domain-template-dec.pgmx.j2" if not PUSH_PEN else "bp-domain-template-dec-badpen.pgmx.j2"}
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
