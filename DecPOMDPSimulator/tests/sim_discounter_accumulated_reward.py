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
all_idle_action: Callable[[Any], int] = lambda ja: all(['action_idle' == a for a in ja])

# num_rocks = 8
# is_final_state: Callable[[Any], bool] = lambda s: all(
#     [s['rock%d'% rock_num] == 'bad' for rock_num in range(1, num_rocks + 1)])
# num_boxes: Callable[[Any], int] = lambda s: sum(
#     [s['rock%d' % rock_num] == 'good' for rock_num in range(1, num_rocks + 1)])

NUM_RUNS = 1000
horizons = [200]

for horizon in horizons:
    num_wins = 0
    Simulator = SimulatorFactory.create_simulator(policies_paths=policies_paths, problem_path=problem_path,
                                                  horizon=horizon,
                                                  problem_kind=problem_kind)
    total_reward = 0
    run_num = 0
    nonempty_runs = 0
    max_step_in_win = 0
    steps_in_winning_games = 0

    while run_num < NUM_RUNS:
        Simulator.reset()

        nonempty_runs += (initial_num_boxes(Simulator.state) != 0)
        step = 0

        for step in range(horizon):
            step += 1
            Simulator.tick()
            total_reward += (discount ** (step - 1)) * Simulator.last_reward

            if is_final_state(Simulator.state) and all_idle_action(Simulator.last_joint_action):
                max_step_in_win = max(step, max_step_in_win)
                num_wins += 1
                steps_in_winning_games += step
                break

        run_num += 1

    with_empty_avg_reward = total_reward / NUM_RUNS
    without_empty_avg_reward = total_reward / nonempty_runs
    print("Won %d out of %d nonempty games, %d percent" % (num_wins, nonempty_runs, (100 * num_wins / nonempty_runs)))
    print("With empty states, avg accumulated discounted reward: %f" % with_empty_avg_reward)
    print("Without empty states, avg accumulated discounted reward: %f" % without_empty_avg_reward)
    print("Max steps in win: %d" % max_step_in_win)
    avg_steps_for_win = steps_in_winning_games / num_wins if num_wins > 0 else 0
    print("Avg steps in win: %d" % avg_steps_for_win)
