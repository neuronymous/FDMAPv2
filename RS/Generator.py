from functools import reduce
from itertools import combinations
from random import randint
import os

import numpy as np
from jinja2 import Environment, FileSystemLoader

from POMDPX import POMDPXConstants
from RS import RockSamplingConstants
from RS.RockSamplingConstants import DIRECTION_SYMBOLS

""" Note, tile stands for (i,j) base-1 index, where i is the row, j is the column
pos index stands for the linear index of the tile, base-0, which depends on the width as well
"""
# Configuration

# Environment Parameters
PLACE_HOLDER_SYMBOL = 'P'


def actions_per_agent(num_agents, rocks_positions, control_areas):
    cars_actions = []
    for i in range(num_agents):
        car_actions = []
        car_rocks = [j for j, pos in enumerate(rocks_positions) if pos in control_areas[i]]
        for d in DIRECTION_SYMBOLS:
            car_actions.append(RockSamplingConstants.move_action(d, i))
        for j in car_rocks:
            car_actions.append(RockSamplingConstants.sense_action(j, i))

        car_actions.append(RockSamplingConstants.shared_sample_action(i))
        car_actions.append(RockSamplingConstants.private_sample_action(i))
        car_actions.append(RockSamplingConstants.idle_action())

        cars_actions.append(car_actions)
    return cars_actions


def calculate_action_list(control_areas, rocks_positions):
    num_rocks = len(rocks_positions)
    num_cars = len(control_areas)
    all_actions = []

    for i in range(num_cars):
        for d in RockSamplingConstants.DIRECTION_SYMBOLS:
            all_actions.append(RockSamplingConstants.move_action(d, i))
        all_actions += [RockSamplingConstants.private_sample_action(i), RockSamplingConstants.shared_sample_action(i)]
        for j in range(num_rocks):
            if rocks_positions[j] in control_areas[i]:
                all_actions.append(RockSamplingConstants.sense_action(j, i))
    all_actions.append(RockSamplingConstants.idle_action())
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


def calculate_transitions_vec(src_idx, dst_idx, success_prob, num_tiles):
    res = np.zeros(num_tiles)
    res[src_idx] = round(1 - success_prob, 2)
    res[dst_idx] = round(success_prob, 2)
    return [str(i) for i in res]


def calculate_all_tiles(w, h):
    return [(i, j) for j in range(0, w) for i in range(0, h)]


def calculate_pairs(num_elems):
    return [(i, j) for i, j in combinations(range(1, num_elems + 1), 2)]


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


def calculate_direction_matrices(w, h, succ_prob):
    def calculate_dest(src, direction):
        i, j = src[0], src[1]
        if direction == RockSamplingConstants.left():
            j -= 1
        elif direction == RockSamplingConstants.right():
            j += 1
        elif direction == RockSamplingConstants.up():
            i -= 1
        else:
            i += 1

        return (i, j) if 0 <= i <= h - 1 and 0 <= j <= w - 1 else None

    res = {}
    tiles = calculate_all_tiles(w, h)
    num_tiles = w * h
    for d in RockSamplingConstants.DIRECTION_SYMBOLS:
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
        res[d] = cur_matrix
    return res


def calculate_euclidian_distance_in_grid(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const, sense_prob):
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
            succ_prob = (0.5 + 0.5 / (sense_decay_const ** distance_to_rock)) * sense_prob
            curr_mat[curr_pos][RockSamplingConstants.good_idx] = succ_prob
            curr_mat[curr_pos][RockSamplingConstants.bad_idx] = 1 - succ_prob
            curr_mat[curr_pos][RockSamplingConstants.null_idx] = 0
        res[RockSamplingConstants.rock_symbol(rock_idx)] = curr_mat
    return res


def calculate_bad_sense_matrices(w, h, rock_positions, sense_decay_const, sense_prob):
    res = calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const, sense_prob)
    good, bad = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx
    for rock, sense_matrix in res.items():
        sense_matrix[:, [good, bad]] = sense_matrix[:, [bad, good]]
    return res


def calculate_sense_martices(w, h, rock_positions, sense_decay_const, sense_prob):
    res = {}
    good_matrices = calculate_good_sense_matrices(w, h, rock_positions, sense_decay_const, sense_prob)
    bad_matrices = calculate_bad_sense_matrices(w, h, rock_positions, sense_decay_const, sense_prob)
    for rock_idx in range(len(rock_positions)):
        rock_symbol = RockSamplingConstants.rock_symbol(rock_idx)
        res[rock_symbol] = {RockSamplingConstants.good_quality(): good_matrices[rock_symbol],
                            RockSamplingConstants.bad_quality(): bad_matrices[rock_symbol]}
    return res


def calculate_shifted_sense_matrices(w, h, rock_position, sense_decay_const, sense_prob):
    partitions = [[pos for pos in rock_position if pos < w * h],
                  [pos - w + 1 for pos in rock_position if pos - w + 1 < w * h]]

    return [calculate_sense_martices(w, h, shifted_rock_positions, sense_decay_const, sense_prob) for
            shifted_rock_positions in partitions]


def calculate_rock_sense_matrices(w, h, control_area_shift, rock_position, sense_decay_const, sense_prob,
                                  shared_area_width=1):
    good, bad, null = RockSamplingConstants.good_idx, RockSamplingConstants.bad_idx, RockSamplingConstants.null_idx

    all_tiles = [calculate_tile_from_idx(w, h, i) for i in range(w * h)]
    distances = {(i, j): calculate_euclidian_distance_in_grid(all_tiles[i], all_tiles[j]) for i, j in
                 combinations(range(w * h), 2)}

    rock_position -= control_area_shift * (w - shared_area_width)

    while rock_position >= w * h:
        rock_position -= (w - shared_area_width)

    good_matrix = np.zeros(shape=(w * h, 3))
    for curr_pos in range(w * h):
        try:
            distance_to_rock = distances[rock_position, curr_pos]
        except KeyError:
            try:
                distance_to_rock = distances[curr_pos, rock_position]
            except KeyError:
                distance_to_rock = 0
        succ_prob = (0.5 + 0.5 / (sense_decay_const ** distance_to_rock)) * sense_prob
        good_matrix[curr_pos][good] = succ_prob
        good_matrix[curr_pos][bad] = 1 - succ_prob
        good_matrix[curr_pos][null] = 0

    bad_matrix = good_matrix.copy()
    bad_matrix[:, [good, bad]] = good_matrix[:, [bad, good]]
    return {RockSamplingConstants.good_quality(): good_matrix,
            RockSamplingConstants.bad_quality(): bad_matrix}


def calculate_target_indices(target_tiles, w):
    return [calculate_tile_index(w, tile) for tile in target_tiles]


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


def get_identity_matrix(n):
    return np.eye(n)


def full_sample_matrix(i, rock_positions, control_areas, cars_actions, sample_succ_prob):
    """Pr(rocki_1|rocki_0, car1_0, ..., carm_0, car1_action, ... carm_action"""
    relevant_cars = [j for j, c in enumerate(control_areas) if rock_positions[i] in c]
    m = len(relevant_cars)
    rock_pos = rock_positions[i]
    res = np.zeros(shape=[2, 2, *[len(control_areas[j]) for j in relevant_cars],
                          *[len(cars_actions[j]) for j in relevant_cars]])

    is_shared_rock = m > 1
    sample_action_identifier = RockSamplingConstants.is_shared_sample_action if is_shared_rock \
        else RockSamplingConstants.is_private_sample_action

    relevant_sample_actions_indices_per_car = {i: [idx for idx, action in enumerate(cars_actions[i]) if
                                                   sample_action_identifier(action)] for i in relevant_cars}
    for index in np.ndindex(res.shape):
        rock_1 = index[0]
        rock_0 = index[1]
        was_good = rock_0 == 0
        is_good = rock_1 == 0
        curr_prob = 0.0
        if not was_good:
            if is_good:
                curr_prob = 0.0
            else:
                curr_prob = 1.0

        elif was_good:
            cars_in_pos = [j for idx, j in enumerate(relevant_cars) if control_areas[j][index[2 + idx]] == rock_pos]
            if len(cars_in_pos) == 0:
                if is_good:
                    curr_prob = 1.0
                else:
                    curr_prob = 0.0
            else:
                curr_prob = 1 - reduce(lambda x, y: x * y,
                                       [1.0, *[(1 - sample_succ_prob) for idx, i in enumerate(cars_in_pos) if
                                               index[2 + m + idx] in relevant_sample_actions_indices_per_car[i]]])
                if is_good:
                    curr_prob = 1 - curr_prob

        res[index] = curr_prob
    np.set_printoptions(threshold=np.inf)
    return ' '.join(list(map(str, res.flatten('F'))))


def generate_rock_rewards_matrix(rocks_positions, valid_positions, reward):
    """r1_0, r2_0,...,rn_0 r1_1, ... rn_1"""
    n = sum([p in valid_positions for p in rocks_positions])
    res = np.zeros(shape=[*[2 for _ in range(n)], *[2 for _ in range(n)]])
    for index in np.ndindex(res.shape):
        zero_state = [index[i] % 2 for i in range(n)]
        one_state = [index[i] % 2 for i in range(n, 2 * n)]
        if sum(one_state) == n and sum(zero_state) < n:
            res[index] = reward
    np.set_printoptions(threshold=np.inf)
    return ' '.join(list(map(str, res.flatten('F'))))


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
                       calculate_neighbors_dict=calculate_neighbors_dict,
                       calculate_direction_matrices=calculate_direction_matrices,
                       calculate_rock_sense_matrices=calculate_rock_sense_matrices,
                       calculate_target_indices=calculate_target_indices,
                       rocksamping_constants=RockSamplingConstants,
                       pomdpx_constants=POMDPXConstants,
                       time_zero=POMDPXConstants.time_zero,
                       time_one=POMDPXConstants.time_one,
                       get_rock_sample_reward_matrix=get_rock_sample_reward_matrix,
                       project_sense_matrix_to_control_area=project_sense_matrix_to_control_area,
                       project_direction_matrix_to_control_area=project_direction_matrix_to_control_area,
                       matrix_to_string=matrix_to_string,
                       calculate_sample_matrix=calculate_sample_matrix,
                       actions_per_agent=actions_per_agent,
                       get_identity_matrix=get_identity_matrix,
                       generate_rock_rewards_matrix=generate_rock_rewards_matrix,
                       pgmx_sample_matrix=full_sample_matrix)
    template = env.get_template(template_file_name)
    return template.render(parameters)


def create_control_areas(width, height):
    assert width % 2 == 1
    assert height % 2 == 1
    half_width = int(width / 2 + 1)
    half_height = int(height / 2 + 1)
    control_areas = [[i + j * width for i in range(half_width) for j in range(half_height)],
                     [i + j * width + (width - half_width) for i in range(half_width) for j in range(half_height)],
                     [i + j * width + width * (height - half_height) for i in range(half_width) for j in
                      range(half_height)],
                     [i + j * width + width * (height - half_height) + (width - half_width) for i in range(half_width)
                      for j in
                      range(half_height)]]

    return control_areas


def generate_domain_config(width, height, num_cars, num_private_rocks, num_shared_rocks, discount=0.99,
                           prob_move=1.0, prob_sample=1.0, sense_exp_decay=0.8, move_cost=0.5, sense_cost=1.0,
                           endstate_reward=10.0, bad_sample_penalty=20.0,
                           domain_name_suffix='',
                           cars_positions=None, private_rocks_positions=None, shared_rocks_positions=None,
                           control_areas=None, bias_constant=0.0, rocks_goodness_probs=None, good_rock_prob=0.5):
    res = {"DOMAIN_NAME": "RS-%dx%d_%dC_%dS_%dP_%s" % (
        width, height, num_cars, num_shared_rocks, num_private_rocks, domain_name_suffix),
           "NUM_ROCKS": num_shared_rocks + num_private_rocks,
           "NUM_SHARED_ROCKS": num_shared_rocks,
           "NUM_PRIVATE_ROCKS": num_private_rocks,
           "NUM_CARS": num_cars,
           "CARS": [RockSamplingConstants.car_symbol(i) for i in range(num_cars)],
           "ROCKS": [RockSamplingConstants.rock_symbol(i) for i in range(num_private_rocks + num_shared_rocks)],
           "WIDTH": width,
           "HEIGHT": height,
           "DISCOUNT": discount,
           "CARS_POSITIONS": cars_positions,
           "SHARED_ROCKS_POSITIONS": shared_rocks_positions,
           "PRIVATE_ROCKS_POSITIONS": private_rocks_positions,
           "ROCKS_POSITIONS": private_rocks_positions + shared_rocks_positions,
           "CONTROL_AREAS": control_areas,
           "ROCKS_GOOD_PROBS": [good_rock_prob for _ in range(
               num_private_rocks + num_shared_rocks)] if rocks_goodness_probs is None else rocks_goodness_probs,
           "PROB_MOVE": prob_move,
           "PROB_SAMPLE": prob_sample,
           "SENSE_DECAY_CONST": sense_exp_decay,
           "MOVE_COST": -move_cost,
           "SENSE_COST": -sense_cost,
           "BAD_SAMPLE_PENALTY": -bad_sample_penalty,
           "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS,
           "CARS_BIAS": [bias_constant ** i for i in range(num_cars)],
           "CARS_SYMBOLS": [RockSamplingConstants.car_symbol(i) for i in range(num_cars)],
           "ROCKS_SYMBOLS": [RockSamplingConstants.rock_symbol(i) for i in
                             range(num_private_rocks + num_shared_rocks)],
           "TARGET_REWARD": endstate_reward,
           "PROB_SENSE": sense_prob,
           "SAMPLE_COST": -sample_cost,
           "GOOD_SAMPLE_REWARD": clear_controlarea_reward

           }
    return res


width, height = 3, 4
easy = False
toy = True
half_width = int(width / 2 + 1)
half_height = int(height / 2 + 1)
control_areas = [[i + j * width for j in range(height) for i in range(half_width)],
                 [i + j * width + (width - half_width) for j in range(height) for i in range(half_width)]]
private_tiles_indices = [i for i in range(width * height) if
                         sum([i in control_area for control_area in control_areas]) == 1]
shared_tiles_indices = [i for i in range(width * height) if i not in private_tiles_indices]

if width == 7:
    if not easy:
        private_rocks_tiles = [(0, 2), (1, 2), (3, 4), (2, 4), (2, 6), (1, 0)]
    else:
        private_rocks_tiles = [(0, 2), (3, 4), (2, 6), (1, 0)]
    shared_rocks_tiles = [(0, 3), (3, 3)]
    cars_tiles = [(0, 0), (3, 6)]
elif width == 9:
    if not easy:
        private_rocks_tiles = [(0, 2), (3, 4), (2, 6), (1, 0)]
    else:
        private_rocks_tiles = []
    shared_rocks_tiles = [(0, 4), (3, 4)]
    cars_tiles = [(0, 0), (3, 8)]


elif width == 5:
    if not easy:
        private_rocks_tiles = [(0, 1), (3, 3), (2, 4), (1, 0)]
    else:
        private_rocks_tiles = [(0, 1), (3, 3)]
    shared_rocks_tiles = [(0, 2), (3, 2)]
    cars_tiles = [(0, 0), (3, 4)]
elif width == 3:
    if height == 2:
        private_rocks_tiles = []
        shared_rocks_tiles = [(0, 1), (1, 1)]
        cars_tiles = [(0, 0), (1, 2)]

    elif height == 4:
        if toy:
            private_rocks_tiles = []
            shared_rocks_tiles = [(3, 1), (0,1)]
            cars_tiles = [(0, 0), (3, 2)]
        else:
            private_rocks_tiles = [(2, 2), (1, 0)]
            shared_rocks_tiles = [(0, 1), (3, 1)]
            cars_tiles = [(0, 0), (3, 2)]
else:
    raise Exception("Width must be 3 5 or 7")

private_rocks_positions = [calculate_tile_index(width, tile) for tile in private_rocks_tiles]
shared_rocks_positions = [calculate_tile_index(width, tile) for tile in shared_rocks_tiles]
cars_positions = [{calculate_tile_index(width, cars_tiles[i]): 1.0} for i in range(2)]

num_private_rocks, num_shared_rocks = len(private_rocks_positions), len(shared_rocks_positions)
assert all([pos in private_tiles_indices for pos in private_rocks_positions])
assert all([pos in shared_tiles_indices for pos in shared_rocks_positions])
clear_controlarea_reward = 750
bad_sample_penalty = 500
sense_prob = 0.9
good_rock_prob = 0.5
move_cost = 5
sense_cost = 1
sample_cost = 0.5
sense_exp_decay = 2.5
bias_constant = 1.0

templates = ["rs-domain-template.pomdpx.j2", "rs-domain-template-minobs.pomdpx.j2"]
domains = {
    "initial": {"config": generate_domain_config(width=width, height=height, num_cars=2,
                                                 endstate_reward=clear_controlarea_reward,
                                                 bad_sample_penalty=bad_sample_penalty, move_cost=move_cost,
                                                 sense_cost=sense_cost,
                                                 sense_exp_decay=sense_exp_decay,
                                                 cars_positions=cars_positions,
                                                 control_areas=control_areas,
                                                 bias_constant=bias_constant,
                                                 domain_name_suffix="TEAM",
                                                 num_private_rocks=num_private_rocks,
                                                 num_shared_rocks=num_shared_rocks,
                                                 private_rocks_positions=private_rocks_positions,
                                                 shared_rocks_positions=shared_rocks_positions,
                                                 good_rock_prob=good_rock_prob),
                "template": "rs-domain-template-team.pomdpx.j2"},
    "initial2": {"config": generate_domain_config(width=width, height=height, num_cars=2,
                                                  endstate_reward=clear_controlarea_reward,
                                                  bad_sample_penalty=bad_sample_penalty, move_cost=move_cost,
                                                  sense_cost=sense_cost,
                                                  sense_exp_decay=sense_exp_decay,
                                                  cars_positions=cars_positions,
                                                  control_areas=control_areas,
                                                  bias_constant=bias_constant,
                                                  domain_name_suffix="DEC",
                                                  num_private_rocks=num_private_rocks,
                                                  num_shared_rocks=num_shared_rocks,
                                                  private_rocks_positions=private_rocks_positions,
                                                  shared_rocks_positions=shared_rocks_positions,
                                                  good_rock_prob=good_rock_prob),
                 "template": "rs-domain-template-team.pomdpx.j2"},
    "initial3": {"config": generate_domain_config(width=width, height=height, num_cars=2,
                                                  endstate_reward=clear_controlarea_reward,
                                                  bad_sample_penalty=bad_sample_penalty, move_cost=move_cost,
                                                  sense_cost=sense_cost,
                                                  sense_exp_decay=sense_exp_decay,
                                                  cars_positions=cars_positions,
                                                  control_areas=control_areas,
                                                  bias_constant=bias_constant,
                                                  domain_name_suffix="DEC",
                                                  num_private_rocks=num_private_rocks,
                                                  num_shared_rocks=num_shared_rocks,
                                                  private_rocks_positions=private_rocks_positions,
                                                  shared_rocks_positions=shared_rocks_positions,
                                                  good_rock_prob=good_rock_prob),
                 "template": "rs-domain-template-dec.pgmx.j2"}
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
