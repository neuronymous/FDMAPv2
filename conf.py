import os


class Delegator(type):
    def __getattribute__(self, item):
        try:
            return getattr(Config, item)
        except:
            return object.__getattribute__(self, item)


class Config(object):
    random_seed = 3131341511581881
    team_precision = 25
    sa_precision = 50
    resources_dir = os.path.join(os.path.dirname(__file__), "Resources")
    solvers_dir = os.path.join(os.path.dirname(__file__), "CPPSolvers")

    cygwin_bash_path = os.path.join("D://Softwares//Cygwin//bin//bash.exe")

    repeat_collab = True

    problems_dir = os.path.join(resources_dir, "problems")
    traces_dir = os.path.join(resources_dir, "traces")
    graphs_dir = os.path.join(resources_dir, "graphs")
    logs_dir = os.path.join(resources_dir, "logs")

    # manual pipeline run params
    problem_name = "RS-7x4_2C_2S_4P_TEAM"

    problem_kind = "RockSampling"
    num_traces = 150
    type_trace_sampling = "heuristic"  # "heuristic", "cb", "iterative"

    trace_format = '.trace'
    problem_format = '.pomdpx'
    graph_format = ".dot"

    sarsop_dir = os.path.join(solvers_dir, "sarsop", "src")
    sarsop_solve_team = os.path.join(sarsop_dir, "solve_team.txt")
    # sarsop_solve_single = os.path.join(sarsop_dir, "solve_single.txt")
    sarsop_solve_single = os.path.join(sarsop_dir, "solve_single_no_depth_limit.txt")

    projector = os.path.join(os.path.dirname(__file__), "TeamProblemProjector", "Projector.py")
    aligner = os.path.join(os.path.dirname(__file__), "GraphAlignment", "GraphAligner.py")
    simulations_dir = os.path.join(os.path.dirname(__file__), "DecPOMDPSimulator", "simulations")

    simulation_problems_dir = os.path.join(simulations_dir, "problems")
    simulation_policies_dir = os.path.join(simulations_dir, "policy_graphs")

    # orchestrator params
    num_processes = 1
    pipeline_logs_dir_name = 'pipeline_logs'
    pipeline_team_problem_name_to_kind = {
        # 'RS-3x4_2C_2S_0P_TEAM': problem_kind,
        'RS-3x4_2C_2S_1P_TEAM': problem_kind
        # 'RS-3x4_2C_2S_2P_TEAM': problem_kind
        # 'RS-5x4_2C_2S_2P_TEAM': problem_kind,
        # 'RS-5x4_2C_2S_4P_TEAM': problem_kind
        # "RS-7x4_2C_2S_6P_TEAM": problem_kind
        # "RS-7x4_2C_2S_4P_TEAM": problem_kind
    }

    # pipeline flags
    SOLVE_TEAM = 1
    PROJECT = 1
    SOLVE_SINGLE = 1
    ALIGN = 1
    SIMULATE = 1

    # team and single timeout
    TIMEOUT = 900

    # simulation params
    HORIZONS = [200]
    NUM_RUNS = 1000
    SIMULATIONS = ['sim_all']
