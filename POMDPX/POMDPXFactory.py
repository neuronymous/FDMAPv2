from BP import Constants as BPConstants
from Corridor import CorridorConstants
from POMDPX import POMDPXConstants
from POMDPX.POMDPX import POMDPXProblem
from RS import RockSamplingConstants
from DecTiger import Constants as DTConstants


class POMDPXProblemFactory:
    @staticmethod
    def create_pomdpx_problem(path, kind, **pomdpx_kwargs):
        domain_constants = POMDPXProblemFactory.get_problem_constants_by_kind(kind)
        pomdpx_constants = POMDPXConstants

        problem = POMDPXProblem()
        problem.register_pomdpx_constants(pomdpx_constants)
        problem.register_domain_constants(domain_constants)
        problem.parse_problem(path)
        return problem

    @staticmethod
    def get_problem_constants_by_kind(kind):
        domain_constants = None
        if kind == 'BoxPushing':
            domain_constants = BPConstants
        elif kind == 'RockSampling':
            domain_constants = RockSamplingConstants
        elif kind == 'Corridor':
            domain_constants = CorridorConstants
        elif kind == "DecTiger":
            domain_constants = DTConstants

        return domain_constants
