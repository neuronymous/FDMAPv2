domains = {
    "d1": {
        "DOMAIN_NAME": "BP-2x2_2A_1H_1L",
        "NUM_BOXES": 2,
        "NUM_AGENTS": 2,
        "WIDTH": 2,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [1],
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None],
        "BOXES_POS_DIST": [{(1, 2): 0.33,
                            (1, 1): 0.67},
                           {(1, 2): 0.33,
                            (2, 1): 0.67},
                           ],
        "PROB_MOVE": 0.8,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -7,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d2": {
        "DOMAIN_NAME": "BP-4x4_2A_1H_1L",
        "NUM_BOXES": 2,
        "NUM_AGENTS": 2,
        "WIDTH": 4,
        "HEIGHT": 4,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [1],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None],
        "BOXES_POS_DIST": [None,
                           None],
        "PROB_MOVE": 0.8,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -7,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },

    "d3": {
        "DOMAIN_NAME": "BP-3x3_2A_1H_2L",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 2,
        "WIDTH": 3,
        "HEIGHT": 3,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [1],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None],
        "BOXES_POS_DIST": [None,
                           None,
                           None,
                           None],
        "PROB_MOVE": 0.8,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -7,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },

    "d4": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d5": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_0L",
        "NUM_BOXES": 1,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [1],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(1)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.9,
        "PROB_JPUSH": 0.9,

        "MOVE_COST": -5,
        "PUSH_COST": -7,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d6": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_DOUBLEPUSHCOST",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -20,
        "JPUSH_COST": -10,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d7": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_4xPUSHCOST",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -40,
        "JPUSH_COST": -20,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d8": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_JOINTBONUS",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 1.0,

        "MOVE_COST": -5,
        "PUSH_COST": -40,
        "JPUSH_COST": -2,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d9": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_HIGHREWARD",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.99,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 5000,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d10": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_LOWCOST",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.99,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -2,
        "PUSH_COST": -3,
        "JPUSH_COST": -1.5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d11": {
        "DOMAIN_NAME": "BP-3x2_3A_1H_2L_RELAXEDCOST",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [3],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(i, j): 0.2 for i in range(1, 3) for j in range(1, 4) if i != 1 or j != 1} for k in
                           range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -5,
        "SENSE_COST": -1,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d12": {
        "DOMAIN_NAME": "BP-4x4_3A_0H_3L_EASYINIT",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 4,
        "HEIGHT": 4,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(1, 1): 0.5, (4, 4): 0.5} for _ in range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -10,
        "SENSE_COST": -2,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
    "d13": {
        "DOMAIN_NAME": "BP-3x3_3A_0H_3L_EASYINIT",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 3,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(1, 1): 0.5, (3, 3): 0.5} for _ in range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -10,
        "SENSE_COST": -2,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    },
"d14": {
        "DOMAIN_NAME": "BP-3x2_3A_0H_3L_EASYINIT",
        "NUM_BOXES": 3,
        "NUM_AGENTS": 3,
        "WIDTH": 3,
        "HEIGHT": 2,
        "DISCOUNT": 0.95,
        "TARGET_TILES": [(1, 1)],
        "HEAVY_BOXES": [],
        # DIST conf is index based, X_POS_DIST[i] defines the positions distribution for Xi
        # None yields a uniform distribution
        "AGENTS_POS_DIST": [None,
                            None,
                            None],
        "BOXES_POS_DIST": [{(1, 1): 0.5, (2, 3): 0.5} for _ in range(3)],
        "PROB_MOVE": 1.0,
        "PROB_PUSH": 0.8,
        "PROB_JPUSH": 0.8,

        "MOVE_COST": -5,
        "PUSH_COST": -10,
        "JPUSH_COST": -10,
        "SENSE_COST": -2,

        "GOAL_REWARD": 500,
        "UNGOAL_PENALTY": -10000,

        "PROB_OBS_BOX": 1.0,

        "DIRECTION_SYMBOLS": DIRECTION_SYMBOLS

    }
}
