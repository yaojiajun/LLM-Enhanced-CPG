# -*- coding: utf-8 -*-
"""
GOAL (Unified CVRP/OVRP/VRPB/VRPL/VRPTW decoder, slim & faster)
Copyright (c) 2024-present NAVER
License: CC BY-NC-SA 4.0
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from learning.reformat_subproblems import (
    remove_origin_and_reorder_tensor,
    remove_origin_and_reorder_matrix,
)
from utils.data_manipulation import prepare_data, prepare_routing_data

# Heuristic hook
try:
    from gpt import heuristics_v2 as heuristics
except Exception:
    from gpt import heuristics

ATTENTION_BIAS_decoder1 = False


# ----------------------------
# Flags via problem name
# ----------------------------
def _has_open(name: str) -> bool:
    return "o" in name.lower()


def _has_bc(name: str) -> bool:
    return "b" in name.lower()


def _has_dc(name: str) -> bool:
    return "d" in name.lower()


def _has_tw(name: str) -> bool:
    return "tw" in name.lower()


# ----------------------------
# Subproblem state
# ----------------------------
@dataclass
class UniVRPSubPb:
    problem_name: str
    dist_matrices: Tensor               # (B, N, N, F) ; F>=1, F=0 为距离
    node_demands: Tensor                # (B, N)
    service_times: Optional[Tensor]     # (B, N) or None
    time_windows: Optional[Tensor]      # (B, N, 2) or None
    departure_times: Optional[Tensor]   # (B, T) or None ; T为已走步数+1
    remaining_capacities: Tensor        # (B, 2..T) : [total_cap, ..., cur_cap]
    remaining_backhaul_capacities: Optional[Tensor]
    remaining_distances: Optional[Tensor]
    original_idxs: Tensor               # (B, N)

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


# ----------------------------
# Tours reconstruction
# ----------------------------
def reconstruct_tours(paths: Tensor, via_depots: Tensor, problem_name: str) -> List[List[int]]:
    bs, L = paths.shape
    tours = [[0] for _ in range(bs)]
    for pos in range(1, L):
        nodes = paths[:, pos].tolist()
        via_pos = (via_depots[:, pos]).nonzero().squeeze(-1).tolist()
        for b in via_pos:
            if pos != 1:  # 第一步 via 无意义
                tours[b].append(0)
        for b in range(bs):
            tours[b].append(nodes[b])  # 最后一列为 N-1（虚拟仓库）
    return tours


# ----------------------------
# Heuristic bias
# ----------------------------
def _compute_dynamic_bias_vector(subpb: UniVRPSubPb) -> torch.Tensor:
    device = subpb.dist_matrices.device
    dtype  = subpb.dist_matrices.dtype
    name   = subpb.problem_name.lower()
    has_tw = _has_tw(name)
    has_bc = _has_bc(name)
    has_op = _has_open(name)
    has_dc = subpb.remaining_distances is not None

    B, N, _, _ = subpb.dist_matrices.shape
    travel_nn = subpb.dist_matrices[..., 0]
    cur_row   = travel_nn[:, 0, :]               # 当前点 -> 所有点
    current_distance_matrix = cur_row.unsqueeze(1)  # (B,1,N)

    dem = subpb.node_demands.to(dtype)
    delivery_node_demands = torch.where(dem > 0, dem, torch.tensor(2.0, device=device, dtype=dtype))
    if has_bc:
        pickup_node_demands = torch.where(dem < 0, dem, torch.tensor(-2.0, device=device, dtype=dtype))
    else:
        pickup_node_demands = torch.zeros_like(delivery_node_demands)

    load = subpb.remaining_capacities[:, -1].to(dtype).unsqueeze(1)  # (B,1)

    if has_op:
        delivery_node_demands_open = delivery_node_demands.clone()
        load_open = load.clone()
    else:
        delivery_node_demands_open = torch.zeros_like(delivery_node_demands)
        load_open = torch.zeros_like(load)

    if has_tw and (subpb.time_windows is not None) and (subpb.departure_times is not None):
        time_windows = subpb.time_windows.to(dtype)
        dep = subpb.departure_times[:, -1].to(dtype)
        eta = dep[:, None] + cur_row
        estimated_arrival = eta.unsqueeze(1)      # (B,1,N)
        late = time_windows[..., 1]
        if torch.isinf(late).all() or (late == 0).all():
            time_windows = torch.zeros_like(time_windows)
            estimated_arrival = torch.zeros_like(estimated_arrival)
    else:
        time_windows = torch.zeros((B, N, 2), device=device, dtype=dtype)
        estimated_arrival = torch.zeros((B, 1, N), device=device, dtype=dtype)

    cur_len = (subpb.remaining_distances[:, -1] if has_dc else torch.zeros(B, device=device, dtype=dtype)).unsqueeze(1)

    bias = torch.stack([
        heuristics(
            current_distance_matrix[i],
            delivery_node_demands[i],
            load[i],
            delivery_node_demands_open[i],
            load_open[i],
            time_windows[i],
            estimated_arrival[i],
            pickup_node_demands[i],
            cur_len[i],
        ) for i in range(B)
    ], dim=0).squeeze(1)

    return bias


# ----------------------------
# Helpers for POMO (parallel)
# ----------------------------
def _repeat_subpb(subpb: UniVRPSubPb, K: int) -> UniVRPSubPb:
    """把 batch 维复制 K 倍：B -> B*K，用于并行 POMO。"""
    if K <= 1:
        return subpb

    def rep(x):
        if x is None:
            return None
        B = x.size(0)
        # (B, ...) -> (B,1,...) -> (B,K,...) -> (B*K,...)
        return x.unsqueeze(1).expand(B, K, *x.shape[1:]).reshape(-1, *x.shape[1:])

    return UniVRPSubPb(
        problem_name=subpb.problem_name,
        dist_matrices=rep(subpb.dist_matrices),
        node_demands=rep(subpb.node_demands),
        service_times=rep(subpb.service_times),
        time_windows=rep(subpb.time_windows),
        departure_times=rep(subpb.departure_times),
        remaining_capacities=rep(subpb.remaining_capacities),
        remaining_backhaul_capacities=rep(subpb.remaining_backhaul_capacities),
        remaining_distances=rep(subpb.remaining_distances),
        original_idxs=rep(subpb.original_idxs),
    )


# ----------------------------
# Public API
# ----------------------------
def decode(problem_name: str,
           data: list,
           net: Module,
           beam_size: int = 1,
           knns: int = -1,
           sample: bool = False,
           make_tours: bool = True,
           use_pomo: bool = True,
           pomo_starts: int = 1,
           pomo_mode: str = "parallel"  # "parallel" or "sequential"
           ) -> Tuple[Optional[Tensor], Optional[List[List[int]]]]:
    name = problem_name.lower()
    if len(data) != 11:
        raise ValueError(f"Unsupported data length {len(data)}; only 11-slot unified format is supported.")

    (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
     service_times, time_windows, departure_times0,
     distance_constraints, remaining_distance_constraints, _) = data

    B = dist_matrices.size(0)
    device = dist_matrices.device
    dtype  = dist_matrices.dtype

    remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)
    remaining_backhaul_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)

    if (departure_times0 is None) or (departure_times0.dim() == 2 and departure_times0.size(1) == 0):
        departure_times = torch.zeros(B, 1, dtype=dtype, device=device)
    else:
        departure_times = departure_times0.to(dtype=dtype, device=device)

    service_times = service_times.to(dtype=dtype, device=device) if service_times is not None else None
    time_windows  = time_windows.to(dtype=dtype, device=device)  if time_windows  is not None else None

    remaining_distances = (
        torch.cat([distance_constraints, remaining_distance_constraints], dim=-1)
        if (distance_constraints is not None and remaining_distance_constraints is not None) else None
    )

    N = dist_matrices.size(1)
    original_idxs = torch.arange(N, device=device)[None, :].expand(B, -1)

    base_subpb = UniVRPSubPb(
        problem_name=problem_name,
        dist_matrices=dist_matrices,
        node_demands=node_demands,
        service_times=service_times,
        time_windows=time_windows,
        departure_times=departure_times,
        remaining_capacities=remaining_capacities,
        remaining_backhaul_capacities=remaining_backhaul_capacities,
        remaining_distances=remaining_distances,
        original_idxs=original_idxs,
    )

    if beam_size != 1:
        raise NotImplementedError("Beam search not included.")

    # === 无 POMO：单起点（固定首步为 0） ===
    if (not use_pomo) or (pomo_starts <= 1):
        paths, via_depots, subpb = _greedy_decoding_loop(base_subpb, net, knns, sample)

        # 合法性检查
        sel_sorted = torch.sort(paths[:, 1:-1], dim=1).values
        expect = torch.arange(1, N - 1, device=device).expand_as(sel_sorted)
        assert torch.equal(sel_sorted, expect), "paths contain non-customer indices or duplicates"

        # 可行性检查
        if subpb.remaining_capacities is not None:
            assert (subpb.remaining_capacities >= 0).all()
            if _has_bc(name):
                total_cap = subpb.remaining_capacities[:, 0:1]
                assert (subpb.remaining_capacities <= total_cap + 1e-6).all()
        if subpb.remaining_distances is not None:
            assert (subpb.remaining_distances >= 0).all()

        # 成本
        if make_tours:
            tours = reconstruct_tours(paths, via_depots, problem_name)
            D_all = dist_matrices[..., 0]
            is_open = _has_open(name)
            lens = []
            for b in range(B):
                tour = tours[b]
                D = D_all[b]
                u = torch.tensor(tour[:-1], device=device, dtype=torch.long)
                v = torch.tensor(tour[1:],  device=device, dtype=torch.long)
                seg = D[u, v]
                if is_open:
                    depot_end = (v == 0) | (v == (N - 1))
                    seg = torch.where(depot_end, torch.zeros_like(seg), seg)
                lens.append(seg.sum())
            tour_lens = torch.stack(lens).to(torch.float32)
        else:
            tour_lens, tours = None, None

        return tour_lens, (tours if make_tours else None)

    # === POMO：固定起点 0，仅扩批做 best-of-K（不强制不同首客户） ===
    K = int(max(1, pomo_starts))

    if pomo_mode.lower() == "parallel":
        # 并行：B -> B*K
        subpb_big = _repeat_subpb(base_subpb, K)  # B*K，全部首步都从 0 出发
        BK = B * K

        # 解码（不传 force_first，默认首步全 0）
        paths_big, via_big, subpb_big_out = _greedy_decoding_loop(subpb_big, net, knns, sample)

        # 成本
        D_all = dist_matrices[..., 0]
        is_open = _has_open(name)
        if make_tours:
            tours_big = reconstruct_tours(paths_big, via_big, problem_name)  # len BK
            costs = []
            for i in range(BK):
                # 对并行扩批：第 i 条对应原始样本 b = i % B
                b = i % B
                D = D_all[b]
                tour = tours_big[i]
                u = torch.tensor(tour[:-1], device=device, dtype=torch.long)
                v = torch.tensor(tour[1:],  device=device, dtype=torch.long)
                seg = D[u, v]
                if is_open:
                    depot_end = (v == 0) | (v == (N - 1))
                    seg = torch.where(depot_end, torch.zeros_like(seg), seg)
                costs.append(seg.sum())
            costs = torch.stack(costs).to(torch.float32)  # (BK,)
        else:
            tours_big = None
            costs = torch.zeros(BK, device=device, dtype=torch.float32)

        # (B,K) 取最优
        costs_bk = costs.view(K, B).transpose(0, 1)  # -> (B,K)
        best_k = torch.argmin(costs_bk, dim=1)       # (B,)
        best_cost = costs_bk[torch.arange(B, device=device), best_k]

        if make_tours:
            gather_idx = (best_k * B + torch.arange(B, device=device)).tolist()  # 注意上面 reshape 的排列
            best_tours = [tours_big[i] for i in gather_idx]
            return best_cost, best_tours
        else:
            return best_cost, None

    else:
        # 顺序：跑 K 次，保持起点 0，不指定首客户
        best_cost = torch.full((B,), float("inf"), device=device, dtype=torch.float32)
        best_tours: Optional[List[List[int]]] = None

        for _ in range(K):
            paths_k, via_k, _ = _greedy_decoding_loop(base_subpb, net, knns, sample)

            # 合法性
            sel_sorted_k = torch.sort(paths_k[:, 1:-1], dim=1).values
            expect = torch.arange(1, N - 1, device=device).expand_as(sel_sorted_k)
            if not torch.equal(sel_sorted_k, expect):
                continue

            # 成本
            if make_tours:
                tours_k = reconstruct_tours(paths_k, via_k, problem_name)
                D_all = dist_matrices[..., 0]
                is_open = _has_open(name)
                lens_k = []
                for b in range(B):
                    tour = tours_k[b]
                    D = D_all[b]
                    u = torch.tensor(tour[:-1], device=device, dtype=torch.long)
                    v = torch.tensor(tour[1:],  device=device, dtype=torch.long)
                    seg = D[u, v]
                    if is_open:
                        depot_end = (v == 0) | (v == (N - 1))
                        seg = torch.where(depot_end, torch.zeros_like(seg), seg)
                    lens_k.append(seg.sum())
                cost_k = torch.stack(lens_k).to(torch.float32)
            else:
                tours_k = None
                cost_k = torch.zeros(B, device=device, dtype=torch.float32)

            better = cost_k < best_cost
            if better.any():
                best_cost = torch.where(better, cost_k, best_cost)
                if make_tours:
                    if best_tours is None:
                        best_tours = tours_k
                    else:
                        for b in range(B):
                            if bool(better[b].item()):
                                best_tours[b] = tours_k[b]

        # 兜底回退（仍从 0 出发）
        if torch.isinf(best_cost).any():
            paths, via_depots, _ = _greedy_decoding_loop(base_subpb, net, knns, sample)
            if make_tours:
                tours = reconstruct_tours(paths, via_depots, problem_name)
                D_all = dist_matrices[..., 0]
                is_open = _has_open(name)
                lens = []
                for b in range(B):
                    tour = tours[b]
                    D = D_all[b]
                    u = torch.tensor(tour[:-1], device=device, dtype=torch.long)
                    v = torch.tensor(tour[1:],  device=device, dtype=torch.long)
                    seg = D[u, v]
                    if is_open:
                        depot_end = (v == 0) | (v == (N - 1))
                        seg = torch.where(depot_end, torch.zeros_like(seg), seg)
                    lens.append(seg.sum())
                best_cost = torch.stack(lens).to(torch.float32)
                best_tours = tours
            else:
                best_tours = None

        return best_cost, (best_tours if make_tours else None)



# ----------------------------
# Greedy decoding
# ----------------------------
def _greedy_decoding_loop(subpb: UniVRPSubPb, net: Module, knns: int, sample: bool,
                          force_first: Optional[Tensor] = None):
    B, N, _, _ = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    paths = torch.zeros((B, N), dtype=torch.long, device=device)
    via_depots = torch.zeros((B, N), dtype=torch.bool, device=device)
    paths[:, -1] = N - 1

    for _ in range(1, N - 1):
        idx_selected, via_depot, subpb = _greedy_decoding_step(subpb, net, knns, sample, force_first=force_first)
        # 写入位置：已写入数量即当前位置
        step_pos = (paths != 0).sum(dim=1)
        step_pos = torch.where(step_pos < (N - 1), step_pos, (N - 2))
        paths[torch.arange(B, device=device), step_pos] = idx_selected
        via_depots[torch.arange(B, device=device), step_pos] = via_depot
        force_first = None  # 首步后取消强制

    return paths, via_depots, subpb


def _greedy_decoding_step(subpb: UniVRPSubPb, net: Module, knns: int, sample: bool,
                          force_first: Optional[Tensor] = None):
    scores = _prepare_input_and_forward_pass(subpb, net, knns, force_first=force_first)

    if sample:
        probs = torch.softmax(scores, dim=-1)
        pick = [
            np.random.choice(np.arange(probs.shape[1]), p=prob.detach().cpu().numpy())
            for prob in probs
        ]
        selected_nodes = torch.tensor(pick, device=probs.device)[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)

    idx_selected = torch.div(selected_nodes, 2, rounding_mode="trunc")
    via_depot = (selected_nodes % 2 == 1)

    idx_selected_original = torch.gather(subpb.original_idxs, 1, idx_selected)
    new_subpb, via_depot = _reformat_subproblem_for_next_step(subpb, idx_selected, via_depot)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subpb


# ----------------------------
# Model input + constraints masking
# ----------------------------
def _mask_tw_dc_scores(scores: Tensor,
                       travel_nn: Tensor,
                       time_windows: Optional[Tensor],
                       dep_last: Optional[Tensor],
                       rem_init: Optional[Tensor],
                       rem_last: Optional[Tensor]):
    # TW
    if (time_windows is not None) and (dep_last is not None):
        tw_latest = time_windows[..., 1]
        arrival_direct = dep_last[:, None] + travel_nn[:, 0, :]
        arrival_via = travel_nn[:, -1, :]
        scores[:, 0::2] = scores[:, 0::2].masked_fill(arrival_direct > tw_latest, float("-inf"))
        scores[:, 1::2] = scores[:, 1::2].masked_fill(arrival_via   > tw_latest, float("-inf"))
    # DC
    if rem_last is not None:
        direct_leg = travel_nn[:, 0, :]
        scores[:, 0::2] = scores[:, 0::2].masked_fill(direct_leg > rem_last[:, None], float("-inf"))
    if rem_init is not None:
        via_leg = travel_nn[:, -1, :]
        scores[:, 1::2] = scores[:, 1::2].masked_fill(via_leg > rem_init[:, None], float("-inf"))
    return scores


def _build_and_forward(subpb: UniVRPSubPb, net: Module, D: Tensor, node_demands: Tensor,
                       rem_caps0: Tensor, rem_caps_last: Tensor,
                       service_times: Optional[Tensor], time_windows: Optional[Tensor],
                       dep_last: Optional[Tensor],
                       rem_init: Optional[Tensor], rem_last: Optional[Tensor],
                       problem_name: str, use_tw: bool):
    if use_tw:
        data = [D, node_demands, rem_caps0, rem_caps_last, None,
                service_times, time_windows, dep_last, None, None, None]
        node_features, edge_features, problem_data = prepare_routing_data(data, "cvrptw")
    else:
        if (rem_init is not None) and (rem_last is not None):
            data = [D, node_demands, rem_caps0, rem_caps_last, None,
                    None, None, None, rem_init, rem_last, None]
        else:
            data = [D, node_demands, rem_caps0, rem_caps_last, None,
                    None, None, None, None, None, None]
        node_features, edge_features, problem_data = prepare_data(data, problem_name)

    scores = net(node_features, edge_features, problem_data)
    return scores


def _prepare_input_and_forward_pass(subpb: UniVRPSubPb, net: Module, knns: int,
                                    force_first: Optional[Tensor] = None) -> Tensor:
    name  = subpb.problem_name.lower()
    use_tw = _has_tw(name)
    use_dc = _has_dc(name)

    B, N, _, F = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    # KNN 分支（保持原实现）
    if 0 < knns < N:
        nn_idx = torch.topk(subpb.dist_matrices[:, :-1, 0, 0], k=knns - 1, largest=False, dim=-1).indices
        nn_idx = torch.cat([nn_idx, torch.full([B, 1], N - 1, device=device, dtype=torch.long)], dim=-1)  # (B,K)

        knn_dem = torch.gather(subpb.node_demands, 1, nn_idx)
        knn_dist = torch.gather(
            subpb.dist_matrices, 1, nn_idx[..., None, None].expand(-1, -1, N, F)
        )
        knn_dist = torch.gather(
            knn_dist, 2, nn_idx[:, None, :, None].expand(-1, knns, -1, F)
        )  # (B,K,K,F)

        rem_caps0  = subpb.remaining_capacities[:, 0][..., None]
        rem_capslt = subpb.remaining_capacities[:, -1][..., None]
        if use_tw:
            knn_st  = torch.gather(subpb.service_times, 1, nn_idx)
            tw_dim  = subpb.time_windows.shape[-1]
            knn_tw  = torch.gather(subpb.time_windows, 1, nn_idx[..., None].expand(-1, -1, tw_dim))
            dep_last = subpb.departure_times[:, -1][..., None]
            scores = _build_and_forward(
                subpb, net, knn_dist, knn_dem, rem_caps0, rem_capslt,
                knn_st, knn_tw, dep_last, None, None, "cvrptw", use_tw=True
            )
            travel_kk = knn_dist[..., 0]
            scores = _mask_tw_dc_scores(
                scores, travel_kk, knn_tw, dep_last.squeeze(-1),
                subpb.remaining_distances[:, 0] if use_dc and (subpb.remaining_distances is not None) else None,
                subpb.remaining_distances[:, -1] if use_dc and (subpb.remaining_distances is not None) else None
            )
        else:
            rem_init = subpb.remaining_distances[:, 0][..., None] if use_dc and (subpb.remaining_distances is not None) else None
            rem_last = subpb.remaining_distances[:, -1][..., None] if use_dc and (subpb.remaining_distances is not None) else None
            scores = _build_and_forward(
                subpb, net, knn_dist, knn_dem, rem_caps0, rem_capslt,
                None, None, None, rem_init, rem_last, subpb.problem_name, use_tw=False
            )
            if use_dc and (subpb.remaining_distances is not None):
                travel_kk = knn_dist[..., 0]
                scores = _mask_tw_dc_scores(
                    scores, travel_kk, None, None,
                    subpb.remaining_distances[:, 0], subpb.remaining_distances[:, -1]
                )

        scores_full = torch.full((B, 2 * N), -np.inf, device=device)
        idx2 = torch.zeros([B, 2 * knns], device=device, dtype=torch.long)
        idx2[:, 0::2] = 2 * nn_idx
        idx2[:, 1::2] = 2 * nn_idx + 1
        scores = torch.scatter(scores_full, 1, idx2, scores)

    # Full 分支
    else:
        rem_caps0  = subpb.remaining_capacities[:, 0].unsqueeze(-1)
        rem_capslt = subpb.remaining_capacities[:, -1].unsqueeze(-1)
        rem_init = subpb.remaining_distances[:, 0].unsqueeze(-1) if use_dc and (subpb.remaining_distances is not None) else None
        rem_last = subpb.remaining_distances[:, -1].unsqueeze(-1) if use_dc and (subpb.remaining_distances is not None) else None
        dep_last = subpb.departure_times[:, -1].unsqueeze(-1) if use_tw else None

        scores = _build_and_forward(
            subpb, net, subpb.dist_matrices, subpb.node_demands,
            rem_caps0, rem_capslt,
            subpb.service_times if use_tw else None,
            subpb.time_windows  if use_tw else None,
            dep_last, rem_init, rem_last,
            subpb.problem_name, use_tw=use_tw
        )

        travel_nn = subpb.dist_matrices[..., 0]
        scores = _mask_tw_dc_scores(
            scores, travel_nn, subpb.time_windows if use_tw else None,
            dep_last.squeeze(-1) if use_tw else None,
            subpb.remaining_distances[:, 0] if use_dc and (subpb.remaining_distances is not None) else None,
            subpb.remaining_distances[:, -1] if use_dc and (subpb.remaining_distances is not None) else None
        )

    # 首步强制选择
    is_first_step = subpb.remaining_capacities.size(1) == 2
    if is_first_step and (force_first is not None):
        keep_cols = 2 * force_first  # (B,)
        scores[:, 0::2] = -np.inf
        scores[torch.arange(B, device=scores.device), keep_cols] = scores[torch.arange(B, device=scores.device), keep_cols]

    if ATTENTION_BIAS_decoder1:
        node_bias = _compute_dynamic_bias_vector(subpb)  # (B,N)
        bias_2n = torch.zeros_like(scores)
        bias_2n[:, 0::2] = node_bias
        bias_2n[:, 1::2] = node_bias
        scores = scores + bias_2n

    # 首步禁止 via
    if is_first_step:
        scores[:, 1::2] = -np.inf

    # 禁用虚拟仓与当前点 0
    depot_even = 2 * (N - 1)
    scores[:, depot_even: depot_even + 2] = -np.inf
    scores[:, 0:2] = -np.inf

    return scores


# ----------------------------
# State transition
# ----------------------------
def _reformat_subproblem_for_next_step(subpb: UniVRPSubPb,
                                       idx_selected: Tensor,
                                       via_depot: Tensor) -> Tuple[UniVRPSubPb, Tensor]:
    name = subpb.problem_name.lower()
    has_tw = _has_tw(name)
    has_dc = _has_dc(name)

    B, S, _, _ = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    is_sel = (torch.arange(S, device=device).unsqueeze(0).expand(B, -1) == idx_selected)

    sel_dem = subpb.node_demands[is_sel].unsqueeze(1)  # (B,1)
    next_dem = remove_origin_and_reorder_tensor(subpb.node_demands, is_sel)
    next_idx = remove_origin_and_reorder_tensor(subpb.original_idxs,  is_sel)

    total_cap = subpb.remaining_capacities[:, 0:1]
    prev_cap  = subpb.remaining_capacities[:, -1].unsqueeze(-1)

    dem_fwd   = torch.clamp(sel_dem,  min=0)
    cand_cap  = prev_cap - dem_fwd
    via_depot = via_depot | (cand_cap < 0)
    next_cap  = torch.where(via_depot, total_cap - sel_dem.abs(), cand_cap)
    next_remaining_capacities = torch.cat([subpb.remaining_capacities, next_cap], dim=-1)

    prev_back = subpb.remaining_backhaul_capacities[:, -1].unsqueeze(-1)
    dem_back  = torch.clamp(-sel_dem, min=0)
    cand_back = prev_back - dem_back
    via_depot = via_depot | (cand_back < 0)
    next_backhaul_cap = torch.where(via_depot, total_cap - dem_back, cand_back)
    next_remaining_backhaul_capacities = torch.cat(
        [subpb.remaining_backhaul_capacities, next_backhaul_cap], dim=-1
    )

    if has_tw:
        assert (subpb.service_times is not None) and (subpb.time_windows is not None) and (subpb.departure_times is not None)
        sel_st  = subpb.service_times[is_sel].unsqueeze(1)
        sel_tw  = subpb.time_windows[is_sel].unsqueeze(1)

        direct_travel = subpb.dist_matrices[:, 0, :, 0][is_sel].unsqueeze(-1)
        arrive_direct = subpb.departure_times[:, -1].unsqueeze(-1) + direct_travel
        arrive_direct = torch.maximum(arrive_direct, sel_tw[..., 0])
        depart_direct = arrive_direct + sel_st

        from_depot = subpb.dist_matrices[:, -1, :, 0][is_sel].unsqueeze(-1)
        arrive_via  = torch.maximum(from_depot, sel_tw[..., 0])
        depart_via  = arrive_via + sel_st

        latest_tw = subpb.time_windows[
            torch.arange(B, device=device), idx_selected.squeeze(1), 1
        ].to(arrive_direct.dtype).unsqueeze(-1)

        direct_viol = (arrive_direct > latest_tw).squeeze(-1)
        via_viol    = (arrive_via    > latest_tw).squeeze(-1)
        force_via   = direct_viol & (~via_viol)
        if force_via.any():
            via_depot[force_via] = True

        next_depart = torch.where(via_depot, depart_via, depart_direct)
        next_departure_times = torch.cat([subpb.departure_times, next_depart], dim=-1)
    else:
        next_departure_times = None

    if has_dc:
        assert subpb.remaining_distances is not None
        direct_leg     = subpb.dist_matrices[:, 0,  :, 0][is_sel].unsqueeze(-1)
        from_depot_leg = subpb.dist_matrices[:, -1, :, 0][is_sel].unsqueeze(-1)
        rem_prev = subpb.remaining_distances[:, -1].unsqueeze(-1)
        rem_init = subpb.remaining_distances[:,  0:1]

        rem_dir = rem_prev - direct_leg
        rem_via = rem_init - from_depot_leg
        next_rem_dist = torch.where(via_depot, rem_via, rem_dir)
        next_remaining_distances = torch.cat([subpb.remaining_distances, next_rem_dist], dim=-1)
    else:
        next_remaining_distances = None

    next_dist_matrices = remove_origin_and_reorder_matrix(subpb.dist_matrices, is_sel)
    next_service_times = remove_origin_and_reorder_tensor(subpb.service_times, is_sel) if subpb.service_times is not None else None
    if subpb.time_windows is not None:
        tw_dim = subpb.time_windows.size(-1)
        sel_tw = subpb.time_windows[is_sel].unsqueeze(1)
        rem_tw = subpb.time_windows[~is_sel].reshape(B, -1, tw_dim)[:, 1:, :]
        next_time_windows = torch.cat([sel_tw, rem_tw], dim=1)
    else:
        next_time_windows = None

    new_subpb = UniVRPSubPb(
        problem_name=subpb.problem_name,
        dist_matrices=next_dist_matrices,
        node_demands=next_dem,
        service_times=next_service_times,
        time_windows=next_time_windows,
        departure_times=next_departure_times,
        remaining_capacities=next_remaining_capacities,
        remaining_backhaul_capacities=next_remaining_backhaul_capacities,
        remaining_distances=next_remaining_distances,
        original_idxs=next_idx,
    )
    return new_subpb, via_depot
