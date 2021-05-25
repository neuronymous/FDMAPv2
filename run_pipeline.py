import os
import sys
from subprocess import run, PIPE
from shutil import copyfile

from TeamProblemProjector.TraceAnalysisUtils import TracesNumGenerator
from conf import Config

from TeamProblemProjector.Projector import main as project
from GraphAlignment.GraphAligner import main as align


def my_run(*args, **kwargs):
    if os.name == "posix":
        return run(args, stdout=PIPE, stderr=PIPE, stdin=PIPE)
    else:
        return run(args=[Config.cygwin_bash_path, *args], stdout=PIPE, stderr=PIPE, stdin=PIPE)


def main(team_problem_name=Config.problem_name, problem_kind=Config.problem_kind, projection_suffix="",
         dec_problem_name=None):
    if dec_problem_name is None:
        dec_problem_name = team_problem_name.replace("TEAM", "DEC")
    os.chdir(Config.sarsop_dir)

    if Config.SOLVE_TEAM:
        print("Solving team problem")
        precision = Config.team_precision
        num_traces = TracesNumGenerator(Config.type_trace_sampling).generate()
        if Config.random_seed is not None:
            rc = my_run(Config.sarsop_solve_team, "-p", str(precision), "-r", Config.resources_dir, "-n", str(num_traces), "-s", str(Config.random_seed), team_problem_name)
        else:
            rc = my_run(Config.sarsop_solve_team, "-p", str(precision), "-r", Config.resources_dir, "-n", str(num_traces), team_problem_name)
        if rc.returncode != 0:
            raise RuntimeError(rc)

    trace_path = os.path.join(Config.traces_dir, team_problem_name + Config.trace_format)
    assert os.path.exists(trace_path)

    if Config.PROJECT:
        print("Projecting team problem")
        projected_problem_path_to_precision = project(team_problem_name, problem_kind, projection_suffix)

    if Config.SOLVE_SINGLE:
        single_agent_problems = list(projected_problem_path_to_precision.keys())
        assert all([os.path.exists(p) for p in single_agent_problems])
        for p in single_agent_problems:
            single_agent_problem_name = p.split('/')[-1].split('.')[0]
            precision = projected_problem_path_to_precision[p]
            print("Solving %s with precision %f" % (single_agent_problem_name, precision))
            rc = my_run(
                Config.sarsop_solve_single, "-p", str(precision), "-r", Config.resources_dir, "-t", str(Config.TIMEOUT),
                single_agent_problem_name)
            if rc.returncode != 0:
                raise RuntimeError(rc)

    else:
        single_agent_problems = list(projected_problem_path_to_precision.keys())

    if Config.ALIGN:
        print("Aligning")
        single_agent_policy_graphs = [
            p.replace(Config.problem_format, Config.graph_format).replace(Config.problems_dir, Config.graphs_dir)
            for p in single_agent_problems]
        single_agent_policy_graphs.reverse()
        print(single_agent_policy_graphs)
        assert all([os.path.exists(g) for g in single_agent_policy_graphs])
        aligned_graphs = align(graphs=single_agent_policy_graphs, suffix=projection_suffix,
                               problem_name=team_problem_name,
                               problem_kind=problem_kind, output_name=team_problem_name + "_ALIGNED")
        assert all([os.path.exists(gp) for gp in aligned_graphs])

    if Config.SIMULATE:
        os.chdir(Config.simulations_dir)
        print("Simulating")
        dec_problem_file = dec_problem_name + Config.problem_format
        copyfile(os.path.join(Config.problems_dir, dec_problem_file),
                 os.path.join(Config.simulation_problems_dir, dec_problem_file))
        dec_problem_path = os.path.join(Config.simulation_problems_dir, dec_problem_file)
        sys.path.append(Config.simulations_dir)  # add package to path
        for sim in Config.SIMULATIONS:
            sim_path = os.path.join(Config.simulations_dir, sim)
            sim_class = getattr(__import__(sim), 'Simulation')
            for idx, gp in enumerate(aligned_graphs):
                copyfile(gp, os.path.join(Config.simulation_policies_dir, os.path.basename(gp)))
            sim = sim_class(aligned_graphs, dec_problem_path, problem_kind)
            sim.run(horizons=Config.HORIZONS, num_runs=Config.NUM_RUNS)


if __name__ == "__main__":
    main(*sys.argv[1:])
