from conf import Delegator


class Defaults(metaclass=Delegator):
    problem_name = "RS-7x4_2C_2S_4P_TEAM"
    output_problem = "ALIGNED_" + problem_name
    suffix = ""
    graphs = ["D:\CS\Thesis\Code\Resources\graphs\RS-7x4_2C_2S_4P_TEAM_car1.dot",
              "D:\CS\Thesis\Code\Resources\graphs\RS-7x4_2C_2S_4P_TEAM_car2.dot"]
