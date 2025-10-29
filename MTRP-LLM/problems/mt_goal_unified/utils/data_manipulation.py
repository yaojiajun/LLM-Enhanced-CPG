# -*- coding: utf-8 -*-
"""
GOAL Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
import numpy as np
from torch import Tensor


# -----------------------------
# helpers
# -----------------------------
def _is_vrp_family(name: str) -> bool:
    n = name.lower()
    return ("vrp" in n) or (n in ["cvrp", "dcvrp", "cvrptw"])


# -----------------------------
# prepare_batch: 仅 VRP 家族（统一 11 段“超集”）
# -----------------------------
def prepare_batch(data, problem, device, sample=True):
    """
    仅支持 VRP 家族，统一返回 11 段“超集”，缺省字段为 None：
    [dist, demands, total_cap, remaining_cap, via_depots,
     service_times|None, time_windows|None, departure_times|None,
     distance_constraints|None, remaining_distance_constraints|None,
     optimal_values]
    """
    ks = "_s" if sample else ""
    name = problem.lower()
    if not _is_vrp_family(name):
        raise NotImplementedError(f"Only VRP family is supported, got: {problem}")

    # 基础字段（一定存在）
    dist_matrices = data[f"dist_matrices{ks}"].to(device)
    node_demands = data[f"node_demands{ks}"].to(device)
    total_capacities = data[f"total_capacities{ks}"].to(device)
    remaining_capacities = data[f"remaining_capacities{ks}"].to(device)
    via_depots = data.get(f"via_depots{ks}", None)
    if via_depots is not None:
        via_depots = via_depots.to(device)
    optimal_values = data[f"tour_lens{ks}"].to(device)

    # 可选 TW
    service_times = time_windows = departure_times = None
    if f"service_times{ks}" in data and f"time_windows{ks}" in data:
        service_times = data[f"service_times{ks}"].to(device)
        time_windows = data[f"time_windows{ks}"].to(device)
        departure_times0 = data.get(f"departure_times{ks}", None)
        if departure_times0 is None:
            B = dist_matrices.shape[0]
            departure_times = torch.zeros(B, 1, dtype=dist_matrices.dtype, device=device)
        else:
            departure_times0 = departure_times0.to(device)
            if departure_times0.dim() == 2 and departure_times0.size(1) == 0:
                B = dist_matrices.shape[0]
                departure_times = torch.zeros(B, 1, dtype=dist_matrices.dtype, device=device)
            else:
                departure_times = departure_times0

    # 可选距离约束
    distance_constraints = remaining_distance_constraints = None
    if f"distance_constraints{ks}" in data and f"remaining_distance_constraints{ks}" in data:
        distance_constraints = data[f"distance_constraints{ks}"].to(device)
        remaining_distance_constraints = data[f"remaining_distance_constraints{ks}"].to(device)

    # 统一“超集”返回（长度为 11）
    return [
        dist_matrices, node_demands, total_capacities, remaining_capacities, via_depots,
        service_times, time_windows, departure_times,
        distance_constraints, remaining_distance_constraints,
        optimal_values,
    ]


# -----------------------------
# prepare_data 路由（仅 VRP）
# -----------------------------
def prepare_data(data, problem, subproblem_size=None):
    name = problem.lower()
    if not _is_vrp_family(name):
        raise NotImplementedError(f"Only VRP family is supported, got: {problem}")
    if subproblem_size is None:
        subproblem_size = 0
    return prepare_routing_data(data, problem, subproblem_size)


# -----------------------------
# prepare_routing_data: 统一 VRP 超集
# -----------------------------
def prepare_routing_data(data, problem, subproblem_size=0) -> tuple[Tensor, Tensor, dict]:
    # 超集拆包（按 prepare_batch 的顺序）
    (dist_all, node_demands, total_caps, remaining_caps_full, via_depots,
     service_times, time_windows, departure_times,
     distance_constraints, remaining_distance_constraints,
     _) = data

    B, N, _, F = dist_all.shape

    # 子图裁剪
    dist_matrices   = dist_all[:, subproblem_size:, subproblem_size:]           # (B, Nsp, Nsp, F)
    node_demands_sp = node_demands[:, subproblem_size:]                         # (B, Nsp)
    remaining_caps_sp = remaining_caps_full[:, subproblem_size]                 # (B,)
    if via_depots is not None:
        via_depots = via_depots[:, subproblem_size:]

    # backhaul 剩余容量（初始化与主容量一致；推进逻辑在 decoder 中完成）
    remaining_backhaul_caps_sp = remaining_caps_sp

    # 自动判定 has_tw / has_dc（按字段是否存在）
    has_tw = (service_times is not None) and (time_windows is not None) and (departure_times is not None)
    has_dc = (distance_constraints is not None) and (remaining_distance_constraints is not None)

    # Edge & Node
    if has_tw:
        Nsp = dist_matrices.size(1)
        time_norm = dist_all.reshape(B, -1).amax(dim=-1).clamp_min(1e-9)
        edge_features = dist_matrices / time_norm[:, None, None, None].repeat(1, Nsp, Nsp, F)

        tw_sp  = time_windows[:, subproblem_size:]       # (B, Nsp, 2)
        dep_sp = departure_times[:, subproblem_size]     # (B,)
        srv_sp = service_times[:, subproblem_size:]      # (B, Nsp)

        tw_norm  = tw_sp / time_norm[:, None, None].repeat(1, Nsp, 2)
        dep_norm = (dep_sp / time_norm)[..., None]
        srv_norm = srv_sp / time_norm[..., None]
        dem_norm = node_demands_sp / total_caps
        cap_norm = (remaining_caps_sp[..., None] / total_caps)

        feats = [
            dem_norm[..., None],
            cap_norm[:, None, :].repeat(1, Nsp, 1),
            srv_norm[..., None],
            tw_norm,
            dep_norm[:, None, :].repeat(1, Nsp, 1),
        ]
        if has_dc:
            dist_total = distance_constraints.squeeze()                 # (B,)
            dist_left  = remaining_distance_constraints[:, 0].squeeze() # (B,)
            dist_ratio = (dist_left / dist_total).clamp(0, 1)[:, None].repeat(1, Nsp)  # (B, Nsp)
            feats.append(dist_ratio[..., None])

        node_features = torch.cat(feats, dim=-1)

        problem_data = {
            "problem_name": problem,
            "travel_times": dist_all[:, subproblem_size:, subproblem_size:],  # 原始矩阵（供掩蔽/检查）
            "node_demands": node_demands_sp,
            "remaining_capacities": remaining_caps_sp,
            "remaining_backhaul_capacities": remaining_backhaul_caps_sp,      # 按要求无条件加入
            "time_windows": tw_sp,
            "departure_times": dep_sp,
            "via_depots": via_depots,
            "loss": "single_cross_entropy",
            "is_multitype": False,
            "seq_len_per_type": None,
        }
        if has_dc:
            problem_data["remaining_distance_constraints"] = remaining_distance_constraints[:, 0].squeeze()
            problem_data["dist_matrices"] = dist_all

    else:
        edge_features = dist_matrices
        dem_norm = node_demands_sp / total_caps
        cap_norm = (remaining_caps_sp[..., None] / total_caps)
        node_features = torch.cat([
            dem_norm[..., None],
            cap_norm[:, None, :].repeat(1, edge_features.shape[1], 1),
        ], dim=-1)

        problem_data = {
            "problem_name": problem,
            "node_demands": node_demands_sp,
            "remaining_capacities": remaining_caps_sp,
            "remaining_backhaul_capacities": remaining_backhaul_caps_sp,      # 按要求无条件加入
            "via_depots": via_depots,
            "loss": "single_cross_entropy",
            "is_multitype": False,
            "seq_len_per_type": None,
        }
        if has_dc:
            problem_data.update({
                "remaining_distance_constraints": remaining_distance_constraints[:, 0].squeeze(),
                "dist_matrices": dist_all,
            })

    return node_features, edge_features, problem_data


# -----------------------------
# create_ground_truth（仅 VRP 家族）
# -----------------------------
def create_ground_truth(bs, problem_data, device):
    name = problem_data["problem_name"].lower()
    if not _is_vrp_family(name):
        raise NotImplementedError(f"create_ground_truth only supports VRP family, got: {name}")

    # 统一 VRP 家族（任意组合），基于 via_depots 的第 2 列
    assert "via_depots" in problem_data, "VRP needs via_depots to build ground truth."
    ground_truth = torch.full((bs,), 2, dtype=torch.long, device=device)
    vd = problem_data["via_depots"]
    if vd is not None and vd.size(1) >= 2:
        ground_truth[vd[:, 1] == 1.0] += 1
    return ground_truth
