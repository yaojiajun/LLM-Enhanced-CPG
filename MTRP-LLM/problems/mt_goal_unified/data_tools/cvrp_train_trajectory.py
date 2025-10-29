"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
import numpy
from scipy.spatial.distance import pdist, squareform

FACTOR = 10000.

def prepare_one_instance(
    coords,
    demands,
    capacity,
    route,
    type,
    cost_from_solver=None,
    distance_constraint=None,  # dcvrp 的全局里程约束（可选）
):
    problem_size = coords.shape[0]

    # 1) 按 0 切分子环
    route_with_subtours, subroute = [], []
    via_depot = [0] * problem_size
    for node_idx in route[1:]:
        if node_idx == 0:
            route_with_subtours.append(subroute)
            subroute = []
        else:
            subroute.append(node_idx)
    route_with_subtours.append(subroute)

    # 2) 以“剩余容量”升序排序子环，并按该顺序构造路径顺序 + 容量轨迹
    route_remaining_capacities = []
    for _route in route_with_subtours:
        tour_capacity = capacity
        for node_idx in _route:
            tour_capacity -= demands[node_idx]
        route_remaining_capacities.append(tour_capacity)
    route_idxs = np.argsort(route_remaining_capacities)

    route_ordered_by_remaining_capacity = [0]
    route_current_capacity = [capacity]
    for num_tour in route_idxs:
        route_ordered_by_remaining_capacity.extend(route_with_subtours[num_tour])
        first = True
        for node in route_with_subtours[num_tour]:
            if first:
                route_current_capacity.append(capacity - demands[node])
                first = False
            else:
                route_current_capacity.append(route_current_capacity[-1] - demands[node])

    # 3) 距离矩阵 & 原始 route 的 cost
    W = squareform(pdist(coords, metric='euclidean'))

    cost = 0.0
    if type == "ocvrp":
        for i in range(len(route) - 1):
            if route[i + 1] != 0:
                cost += W[route[i], route[i + 1]]
    else:  # dcvrp 或其他默认：统计全路径边
        for i in range(len(route) - 1):
            cost += W[route[i], route[i + 1]]

    # if cost_from_solver is not None:
    #     assert numpy.isclose(cost, cost_from_solver, atol=1e-4)
    # 4) 计算与 route_current_capacity 同构且“回仓重置”的 remaining_distance_traj
    remaining_distance_traj = None
    if type == "dcvrp" and distance_constraint is not None:
        # 标量化
        if isinstance(distance_constraint, (list, tuple, np.ndarray)) and np.size(distance_constraint) == 1:
            dc_val = float(np.asarray(distance_constraint).reshape(-1)[0])
        else:
            dc_val = float(distance_constraint)

        remaining_distance_traj = [dc_val]  # 对齐 route_current_capacity[0] 的“起点 = 仓库”
        # 按与你构造 route_ordered_by_remaining_capacity 完全相同的子环顺序循环
        for num_tour in route_idxs:
            prev = 0
            rem = dc_val  # 子环开始处重置
            for node in route_with_subtours[num_tour]:
                rem -= float(W[prev, node])
                remaining_distance_traj.append(rem)
                prev = node

        # 可选：若不希望出现极小负值（数值误差/违规样本），可以进行截断
        # remaining_distance_traj = [max(x, 0.0) for x in remaining_distance_traj]

    # 5) 生成 via_depot 标记（基于容量轨迹）
    route_ordered_by_remaining_capacity.append(0)  # 末尾补回仓索引以便下游使用
    for i in range(1, len(route_current_capacity)):
        if route_current_capacity[i] > route_current_capacity[i - 1]:
            via_depot[i] = 1
    via_depot.append(0)

    # 6) 依据重排顺序重排坐标与需求
    coords_reordered = coords[route_ordered_by_remaining_capacity]
    demands_reordered = demands[route_ordered_by_remaining_capacity]

    # 7) 返回（dcvrp 多一个“逐步剩余里程”序列，其维度与 route_current_capacity 一致）
    if type == "dcvrp":
        return (
            cost,
            coords_reordered,
            demands_reordered,
            route_ordered_by_remaining_capacity,
            route_current_capacity,
            via_depot,
            remaining_distance_traj,
        )
    else:
        return (
            cost,
            coords_reordered,
            demands_reordered,
            route_ordered_by_remaining_capacity,
            route_current_capacity,
            via_depot,
        )
