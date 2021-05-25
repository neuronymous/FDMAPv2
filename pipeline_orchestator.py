from conf import Config
from run_pipeline import main as run_pipeline
from multiprocessing.pool import Pool
from contextlib import redirect_stdout
from datetime import datetime
import os

logs_dir_name = Config.pipeline_logs_dir_name
logs_dir = os.path.join(os.path.dirname(__file__), logs_dir_name)


def create_log_path(team_problem_name):
    return os.path.join(logs_dir, f'{team_problem_name}-pipeline-{datetime.now()}.log')


def solve_problem(team_problem_name, problem_kind):
    log_path = create_log_path(team_problem_name)
    print("solving %s" % str(log_path))
#    with redirect_stdout(log_path):
    run_pipeline(team_problem_name=team_problem_name, problem_kind=problem_kind)


def solve_problems_parallel(team_problem_name_to_kind):
    with Pool(Config.num_processes) as p:
        params_list = list(team_problem_name_to_kind.items())
        print(params_list)
        p.starmap(solve_problem, params_list)


def main(team_problem_name_to_kind=Config.pipeline_team_problem_name_to_kind):
    solve_problems_parallel(team_problem_name_to_kind)


if __name__ == "__main__":
    main()
