from collections import namedtuple
from functools import reduce
from time import sleep
import curses
import os

import numpy

TRACES_DIR = "../Resources/traces"
TRACE_PATH = os.path.join(TRACES_DIR, "./BP-3x2_3A_1H_2L_LOWCOST.trace")
WIDTH = 3
HEIGHT = 2
NUM_AGENTS = 3
NUM_BOXES = 3
HEAVY_BOXES = [1]
INTERVAL = 1
State = namedtuple('State',
                   ['a_%d' % i for i in range(1, NUM_AGENTS + 1)] + ['b_%d' % i for i in range(1, NUM_BOXES + 1)])
CELL_SIZE = 3
scr = curses.initscr()
tile_mapping = {'%d' % i: (int(numpy.floor(i / WIDTH)), int(i % WIDTH)) for i in range(WIDTH * HEIGHT)}
print(tile_mapping)


def state_to_matrix(state):
    res = []
    for i in range(HEIGHT):
        res.append([' ' * CELL_SIZE for i in range(WIDTH)])

    for i in range(NUM_AGENTS):
        cur_agent_pos = state[i]
        x, y = tile_mapping[cur_agent_pos]
        res[x][y] += 'A%d' % (i + 1)

    for i in range(NUM_BOXES):
        cur_box_pos = state[i + NUM_AGENTS]
        x, y = tile_mapping[cur_box_pos]
        if i + 1 in HEAVY_BOXES:
            res[x][y] += 'B%d' % (i + 1)
        else:
            res[x][y] += 'b%d' % (i + 1)
    return res


def print_pretty_matrix(mat):
    res = ""
    max_row_len = max(reduce(lambda x, y: x + y, list(map(lambda x: len(x), row))) + len(row) + 1 for row in mat)
    res += '-' * (max_row_len) + '\n'
    for row in range(len(mat)):
        res += '|'
        for col in range(len(mat[row])):
            res += mat[row][col] + '|'
        res += '\n'
    return res


def main():
    states = []
    with open(TRACE_PATH, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')
            if line.startswith('Y'):
                state_str = line.split(':')[-1]
                state = State(*state_str.strip('()\n').split(','))
                print(state)
                states.append(state)

    for idx, state in enumerate(states):
        state_str = print_pretty_matrix(state_to_matrix(state))
        state_str += 'Step number: %d\n' % idx
        scr.addstr(0, 0, state_str)
        scr.refresh()
        sleep(INTERVAL)


if __name__ == "__main__":
    main()
