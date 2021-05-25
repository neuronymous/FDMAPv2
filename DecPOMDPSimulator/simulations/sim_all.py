from typing import Any, Callable

from DecPOMDPSimulator.Simulator import SimulatorFactory
from POMDPX.POMDPXFactory import POMDPXProblemFactory


class Simulation:
    def __init__(self, policy_graphs, dec_problem_path, problem_kind):
        self.horizons = 0
        self.num_runs = 0
        self.discount = 0.99
        self.output_end_state = False
        self.policy_graphs = policy_graphs
        self.dec_problem_path = dec_problem_path
        self.problem_kind = problem_kind
        problem_constants = POMDPXProblemFactory.get_problem_constants_by_kind(self.problem_kind)

        self.is_final_state = problem_constants.is_final_state
        self.is_idle_action = problem_constants.is_idle_action
        self.all_idle_action: Callable[[Any], int] = lambda ja: all([self.is_idle_action(a) for a in ja])
        self.state_measure = problem_constants.state_measure
        self.is_empty_state = problem_constants.is_empty_state

    def run(self, num_runs=None, horizons=None):
        if num_runs is None:
            num_runs = self.num_runs
        if horizons is None:
            horizons = self.horizons

        for horizon in horizons:
            print("===Start sim for horizon %d===" % horizon)
            num_wins = 0
            num_nonempty_wins = 0
            Simulator = SimulatorFactory.create_simulator(policies_paths=self.policy_graphs,
                                                          problem_path=self.dec_problem_path,
                                                          horizon=horizon,
                                                          problem_kind=self.problem_kind)
            total_reward = 0
            run_num = 0
            nonempty_runs = 0
            max_step_in_win = 0
            steps_in_winning_games = 0

            while run_num < num_runs:
                Simulator.reset()

                is_empty_run = self.is_empty_state(Simulator.state)
                nonempty_runs += not is_empty_run
                step = 0

                while step < horizon:
                    step += 1
                    Simulator.tick()
                    total_reward += (self.discount ** (step - 1)) * Simulator.last_reward

                    if self.is_final_state(Simulator.state) and self.all_idle_action(Simulator.last_joint_action):
                        max_step_in_win = max(step, max_step_in_win)
                        num_nonempty_wins += int(not is_empty_run)
                        num_wins += 1
                        steps_in_winning_games += step
                        break
                if self.output_end_state:
                    print(Simulator.state)
                run_num += 1

            with_empty_avg_reward = total_reward / num_runs
            without_empty_avg_reward = total_reward / nonempty_runs
            print("Won %d out of %d nonempty games, %d percent" % (
                    num_nonempty_wins, nonempty_runs, (100 * num_nonempty_wins / nonempty_runs)))
            print("Won %d out of %d games, %d percent" % (
                    num_wins, num_runs, (100 * num_wins / num_runs)))            
            print("With empty states, avg accumulated discounted reward: %f" % with_empty_avg_reward)
            print("Without empty states, avg accumulated discounted reward: %f" % without_empty_avg_reward)
            print("Max steps in win: %d" % max_step_in_win)
            avg_steps_for_win = steps_in_winning_games / num_wins if num_wins > 0 else 0
            print("Avg steps in win: %d" % avg_steps_for_win)


# policy_graphs = ["./policy_graphs/RS-7x4_2C_2S_4P_TEAM_ALIGNED_exactalign_postv3_repeatcol_pre_car1.dot",
#                "./policy_graphs/RS-7x4_2C_2S_4P_TEAM_ALIGNED_exactalign_postv3_repeatcol_pre_car2.dot"]
# dec_problem_path = "./problems/RS-7x4_2C_2S_4P_DEC.pomdpx"
# problem_kind = "RockSampling"
#
# s = Simulation(policy_graphs=policy_graphs,
#              problem_kind=problem_kind,
#              dec_problem_path=dec_problem_path)
# s.run(num_runs=100, horizons=[200])
