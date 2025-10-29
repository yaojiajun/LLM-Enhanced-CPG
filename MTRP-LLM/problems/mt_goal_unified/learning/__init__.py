"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

# from learning.tsp.decoding import decode as decode_tsp_like
# from learning.pctsp.decoding import decode as decode_pctsp
from learning.cvrp.decoding import decode as decode_cvrp_like
from learning.ovrp.decoding import decode as decode_ovrp_like
# from learning.op.decoding import decode as decode_op
# from learning.cvrptw.decoding import decode as decode_cvrptw
# from learning.mvc.decoding import decode as decode_mvc
# from learning.mis.decoding import decode as decode_mis
# from learning.kp.decoding import decode as decode_kp
# from learning.upms.decoding import decode as decode_upms
# from learning.jssp.decoding import decode as decode_jssp_like
# from learning.mclp.decoding import decode as decode_mclp


decoding_fn = {
    # 'tsp': decode_tsp_like,
    # 'trp': decode_tsp_like,
    # 'sop': decode_tsp_like,

    # VRP 家族（统一用 decode_cvrp_like）
    'cvrp': decode_cvrp_like,
    'sdcvrp': decode_cvrp_like,
    'ocvrp': decode_cvrp_like,
    'dcvrp': decode_cvrp_like,
    'bcvrp': decode_cvrp_like,
    'obcvrp': decode_cvrp_like,
    'odcvrp': decode_cvrp_like,
    'bdcvrp': decode_cvrp_like,
    'obdcvrp': decode_cvrp_like,

    'cvrptw': decode_cvrp_like,
    'ocvrptw': decode_cvrp_like,
    'bcvrptw': decode_cvrp_like,
    'obcvrptw': decode_cvrp_like,
    'dcvrptw': decode_cvrp_like,
    'odcvrptw': decode_cvrp_like,
    'bdcvrptw': decode_cvrp_like,
    'obdcvrptw': decode_cvrp_like,

    # 其他问题
    # 'op': decode_op,
    # 'pctsp': decode_pctsp,
    # 'kp': decode_kp,
    # 'mvc': decode_mvc,
    # 'mis': decode_mis,
    # 'mclp': decode_mclp,
    # 'upms': decode_upms,
    # 'jssp': decode_jssp_like,
    # 'ossp': decode_jssp_like,
}


