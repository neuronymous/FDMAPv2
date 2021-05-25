from DecPOMDPSimulator.PolicyGraph import PolicyGraph

DOT_PATH = "..\..\Resources\graphs\BP-3x2_3A_0H_3L_CHAINMUST_a1_DELUNUSED_GAPFILLERS_DELGOALS_DELSENSE_PROJACT.dot"
GRAPH_FORMAT = "SARSOP"


def test_basic_functions():
    ip = PolicyGraph(DOT_PATH, GRAPH_FORMAT)
    ip.get_action()
    ip.update_by_obs('null,null,null')
