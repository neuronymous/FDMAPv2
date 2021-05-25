from typing import Any, Callable

from DecPOMDPSimulator.Simulator import SimulatorFactory

num_agents = 2
num_boxes = 3
discount = 0.99
policies_paths = ['BPNEW2-33221PV2NH_exactalign_postv3_repeatcol_pre_agent%d.dot' % i for i in range(1, num_agents + 1)]
problem_path = 'BPNEW2-3x3_2A_2H_1L_HARD_DEC.pomdpx'
problem_kind = 'BoxPushing'
is_final_state: Callable[[Any], bool] = lambda s: all([s['box%d' % (i + 1)] == '0' for i in range(num_boxes)])
initial_num_boxes: Callable[[Any], int] = lambda s: sum([s['box%d' % (i + 1)] != '0' for i in range(num_boxes)])

# num_rocks = 8
# is_final_state: Callable[[Any], bool] = lambda s: all(
#     [s['rock%d'% rock_num] == 'bad' for rock_num in range(1, num_rocks + 1)])
# num_boxes: Callable[[Any], int] = lambda s: sum(
#     [s['rock%d' % rock_num] == 'good' for rock_num in range(1, num_rocks + 1)])
NUM_TRIES = 100
horizon = 100
Simulator = SimulatorFactory.create_simulator(policies_paths=policies_paths, problem_path=problem_path, horizon=horizon,
                                              problem_kind=problem_kind)

for _ in range(NUM_TRIES):
    history = []
    success = False
    step = 0
    start_state = Simulator.state
    if initial_num_boxes(start_state) != 3:
        Simulator.reset()
        continue
    for step in range(horizon):
        history.append(Simulator.state)
        Simulator.tick()
        history.append(Simulator.last_joint_action)
        history.append(Simulator.last_joint_obs)
        # print(tuple(Simulator.state.values()))
        if is_final_state(Simulator.state):
            # for h in history:
            # print(h)
            #print("Won %s after %d steps" % (str(start_state), step))
            success = True
            break
    if success and step > 1:
        continue
    elif step > 1:
        print("Found a losing game")
        print("Start state is", tuple(start_state.values()))
        for h in history:
            print(h)
        break
    else:
        print("Started in final")
    Simulator.reset()
