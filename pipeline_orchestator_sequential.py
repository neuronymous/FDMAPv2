from conf import Config
from run_pipeline import main as run_pipeline
from contextlib import redirect_stdout
from datetime import datetime
import os, sys
import itertools

logs_dir_name = Config.pipeline_logs_dir_name
logs_dir = os.path.join(os.path.dirname(__file__), logs_dir_name)


def create_log_path(team_problem_name):
    return os.path.join(logs_dir, f'{team_problem_name}-pipeline-{datetime.now()}')


def solve_problem(team_problem_name, problem_kind):
    log_path = create_log_path(team_problem_name)
    print("==================Pipeline Starts on %s================" % str(log_path))
    try:
        run_pipeline(team_problem_name=team_problem_name, problem_kind=problem_kind)
        print("==================Pipeline Finished on %s================" % str(log_path))
    except Exception as e:
        print("==================Pipeline ERROR on %s: %s================" % (str(log_path), str(e)))
        pass


def solve_problems_sequential(team_problem_name_to_kind):
    for team_problem_name in team_problem_name_to_kind:
        print("==================Pipeline Sequential INIT on %s================" % str(
            create_log_path(team_problem_name)))
    params_list = list(team_problem_name_to_kind.items())
    for params in params_list:
        solve_problem(*params)


def main(team_problem_name_to_kind=Config.pipeline_team_problem_name_to_kind):
    solve_problems_sequential(team_problem_name_to_kind)


if __name__ == "__main__":
    main()
