from typing import Any, Callable

from DecPOMDPSimulator.Simulator import SimulatorFactory

num_agents = 2
num_boxes = 3
discount = 0.99
policies_paths = ['BPNEW2-33221PV2_exactalign_postv3_repeatcol_pre_agent%d.dot' % i for i in range(1, num_agents + 1)]
problem_path = 'BPNEW2-3x3_2A_2H_1L_HARD_DEC.pomdpx'
problem_kind = 'BoxPushing'
is_final_state: Callable[[Any], bool] = lambda s: all([s['box%d' % (i + 1)] == '0' for i in range(num_boxes)])
initial_num_boxes: Callable[[Any], int] = lambda s: sum([s['box%d' % (i + 1)] != '0' for i in range(num_boxes)])

# num_rocks = 8
# is_final_state: Callable[[Any], bool] = lambda s: all(
#     [s['rock%d'% rock_num] == 'bad' for rock_num in range(1, num_rocks + 1)])
# num_boxes: Callable[[Any], int] = lambda s: sum(
#     [s['rock%d' % rock_num] == 'good' for rock_num in range(1, num_rocks + 1)])

NUM_RUNS = 100
horizon = 200
Simulator = SimulatorFactory.create_simulator(policies_paths=policies_paths, problem_path=problem_path, horizon=horizon,
                                              problem_kind=problem_kind)
num_wins = 0
run_num = 0
steps_in_winning_games = 0
max_step_in_win = 0
while run_num < NUM_RUNS:
    Simulator.reset()
    success = False
    step = 0

    if initial_num_boxes(Simulator.state) != 3:
        continue

    for step in range(horizon):
        step += 1
        Simulator.tick()

        if is_final_state(Simulator.state):
            max_step_in_win = max(step, max_step_in_win)
            steps_in_winning_games += step
            success = True
            break
    if success:
        num_wins += 1
    run_num += 1

avg_steps_for_win = steps_in_winning_games / num_wins if num_wins > 0 else 0
print("Won %d games out of %d with average of %d steps and max of %d steps" % (num_wins, NUM_RUNS, avg_steps_for_win, max_step_in_win))
