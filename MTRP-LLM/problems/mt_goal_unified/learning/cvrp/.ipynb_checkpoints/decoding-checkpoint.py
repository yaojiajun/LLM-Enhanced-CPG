"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, List, Tuple
from learning.reformat_subproblems import (
    remove_origin_and_reorder_tensor,
    remove_origin_and_reorder_matrix,
)
from utils.data_manipulation import prepare_data, prepare_routing_data
from utils.misc import compute_tour_lens


# ===== Heuristic hooks (mirror CVRP example) =====
try:
    from gpt import heuristics_v2 as heuristics
except:
    from gpt import heuristics



ATTENTION_BIAS_encoder  = False   # 预留：与示例保持一致（本文件未用）
ATTENTION_BIAS_decoder  = False   # 第二路偏置：decoder 专用（使用 heuristics_decoder）
ATTENTION_BIAS_decoder1 = True    # 第三路偏置：decoder 动态（使用 heuristics）
# =================================================



# ----------------------------
# Flags via problem name
# ----------------------------

def _has_open(name: str) -> bool:   # o: open VRP
    return "o" in name.lower()

def _has_bc(name: str) -> bool:     # b: pickup&delivery (backhauls / signed demands)
    return "b" in name.lower()

def _has_dc(name: str) -> bool:     # d: distance constraint
    return "d" in name.lower()

def _has_tw(name: str) -> bool:     # tw: time windows
    return "tw" in name.lower()


# ----------------------------
# Subproblem state
# ----------------------------

@dataclass
class UniVRPSubPb:
    problem_name: str
    dist_matrices: Tensor
    node_demands: Tensor
    service_times: Optional[Tensor]
    time_windows: Optional[Tensor]
    departure_times: Optional[Tensor]
    remaining_capacities: Tensor
    remaining_distances: Optional[Tensor]
    original_idxs: Tensor
    # --- LLM heuristic biases (cached per-batch, per-step origin is row-0) ---
    attention_bias2: Optional[Tensor] = None  # shape: (B, N, N) or None
    attention_bias3: Optional[Tensor] = None  # shape: (B, N, N) or None

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}



# ----------------------------
# Tours reconstruction
# ----------------------------

def reconstruct_tours(paths: Tensor, via_depots: Tensor, problem_name: str) -> list[list[int]]:
    has_open = _has_open(problem_name)
    bs = paths.shape[0]
    tours = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_to_add = paths[:, pos].tolist()
        via_pos = (via_depots[:, pos]).nonzero().squeeze(-1).cpu().numpy()
        for b in via_pos:
            if pos == 1:  # ignore head-step via depot (corner case)
                continue
            tours[b].append(0)
        for b in range(bs):
            tours[b].append(nodes_to_add[b])
    return tours


# ----------------------------
# Public API
# ----------------------------

def decode(problem_name: str,
           data: list,
           net: Module,
           beam_size: int = 1,
           knns: int = -1,
           sample: bool = False,
           make_tours: bool = True) -> Tuple[Optional[Tensor], Optional[List[List[int]]]]:
    name = problem_name.lower()
    by_name_tw = _has_tw(name)
    by_name_dc = _has_dc(name)

    L = len(data)
    # 允许名字和长度都作为hint，但最终以“字段是否为 None”为准
    data_has_tw = by_name_tw or (L in (9, 11))
    data_has_dc = by_name_dc or (L in (8, 11))

    if L == 11:
        # 可能是 tw + d，也可能是 tw 但多给了占位 None
        (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
         service_times, time_windows, departure_times0,
         distance_constraints, remaining_distance_constraints, _) = data

        bs = dist_matrices.shape[0]
        remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)

        # 出发时间兜底
        if (departure_times0 is None) or (departure_times0.dim() == 2 and departure_times0.size(1) == 0):
            departure_times = torch.zeros(bs, 1, dtype=dist_matrices.dtype, device=dist_matrices.device)
        else:
            departure_times = departure_times0.to(dist_matrices.dtype).to(dist_matrices.device)

        # 统一 dtype/device
        # 统一 dtype/device（仅在非 None 时转换）
        service_times = (service_times.to(dist_matrices.dtype).to(dist_matrices.device)
                                  if service_times is not None else None)
        time_windows = (time_windows.to(dist_matrices.dtype).to(dist_matrices.device)
                                  if time_windows is not None else None)

        # 只有在两个距离张量都不是 None 时，才视为真的 “有 d”
        if (distance_constraints is not None) and (remaining_distance_constraints is not None):
            remaining_distances = torch.cat([distance_constraints, remaining_distance_constraints], dim=-1)
        else:
            remaining_distances = None  # 实际是 “仅 TW”

    elif L == 9:
        # only tw
        (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
         service_times, time_windows, departure_times0, _) = data

        bs = dist_matrices.shape[0]
        remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)
        if (departure_times0 is None) or (departure_times0.dim() == 2 and departure_times0.size(1) == 0):
            departure_times = torch.zeros(bs, 1, dtype=dist_matrices.dtype, device=dist_matrices.device)
        else:
            departure_times = departure_times0.to(dist_matrices.dtype).to(dist_matrices.device)

        service_times = (service_times.to(dist_matrices.dtype).to(dist_matrices.device)
                                  if service_times is not None else None)
        time_windows = (time_windows.to(dist_matrices.dtype).to(dist_matrices.device)
                                  if time_windows is not None else None)
        remaining_distances = None

    elif L == 8:
        # only d
        (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
         distance_constraints, remaining_distance_constraints, _) = data

        remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)
        # 这里 8 段按定义必然有 d
        remaining_distances = torch.cat([distance_constraints, remaining_distance_constraints], dim=-1)
        service_times = None
        time_windows = None
        departure_times = None

    elif L == 6:
        # plain
        dist_matrices, node_demands, total_capacities, remaining_cap0, _, _ = data
        remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)
        service_times = None
        time_windows = None
        departure_times = None
        remaining_distances = None

    else:
        raise ValueError(
            f"Unsupported data length {L} for problem '{problem_name}'. "
            f"Expected one of [6, 8, 9, 11]."
        )

    # 构造 subpb 时，不再用 data_has_dc 判断；直接用 remaining_distances 是否为 None
    bs, num_nodes, _, _ = dist_matrices.shape
    original_idxs = torch.arange(num_nodes, device=dist_matrices.device)[None, :].repeat(bs, 1)

    subpb = UniVRPSubPb(
        problem_name=problem_name,
        dist_matrices=dist_matrices,
        node_demands=node_demands,
        service_times=service_times if service_times is not None else None,
        time_windows=time_windows if time_windows is not None else None,
        departure_times=departure_times if 'departure_times' in locals() else None,
        remaining_capacities=remaining_capacities,
        remaining_distances=remaining_distances,  # 可能是 None
        original_idxs=original_idxs,
    )
    has_tw = subpb.time_windows is not None
    has_dc = subpb.remaining_distances is not None
    if beam_size == 1:
        paths, via_depots, subpb = _greedy_decoding_loop(subpb, net, knns, sample)
    else:
        raise NotImplementedError("Unified beam search is not included yet.")

    # 更可靠的路径合法性校验：每行必须恰好包含 1..N-2 （不含 0 和 N-1）
    sorted_sel = torch.sort(paths[:, 1:-1], dim=1).values
    expect = torch.arange(1, num_nodes - 1, device=paths.device).expand(paths.size(0), -1)
    assert torch.equal(sorted_sel, expect), "paths contain non-customer indices or duplicates"

    # 可行性检查
    if subpb.remaining_capacities is not None:
        assert (subpb.remaining_capacities >= 0).all()
        if _has_bc(name):
            total_cap = subpb.remaining_capacities[:, 0:1]
            assert (subpb.remaining_capacities <= total_cap + 1e-6).all()
    if has_dc:
        assert (subpb.remaining_distances is not None) and ((subpb.remaining_distances >= 0).all())

    if make_tours:
        tours = reconstruct_tours(paths, via_depots, problem_name)
        D_all = dist_matrices[..., 0]  # (B,N,N) 用第0通道距离
        is_open = _has_open(problem_name)

        lens = []
        for b in range(bs):
            tour = tours[b]  # 例如 [0, c1, 0, c2, ..., ck,(可能0)]
            D = D_all[b]  # (N,N) tensor

            # 相邻边长度：D[t[i], t[i+1]]
            u = torch.tensor(tour[:-1], device=D.device, dtype=torch.long)
            v = torch.tensor(tour[1:], device=D.device, dtype=torch.long)
            seg = D[u, v].clone()  # (L-1,)

            if is_open:
                # 只在 OVRP 下：把“以 0 为终点”的边清零（即 a->0 清零；0->b 不变）
                mask_end_at_depot = (v == 0)
                seg[mask_end_at_depot] = 0.0

            lens.append(seg.sum())

        tour_lens = torch.stack(lens).to(torch.float32)
    else:
        tours = None
        tour_lens = None

    return tour_lens, tours


# ----------------------------
# Greedy decoding
# ----------------------------

def _greedy_decoding_loop(subpb: UniVRPSubPb, net: Module, knns: int, sample: bool):
    bs, num_nodes, _, _ = subpb.dist_matrices.shape
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=subpb.dist_matrices.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=subpb.dist_matrices.device)
    paths[:, -1] = num_nodes - 1
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, subpb = _greedy_decoding_step(subpb, net, knns, sample)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot
    return paths, via_depots, subpb


def _greedy_decoding_step(subpb: UniVRPSubPb, net: Module, knns: int, sample: bool):
    scores = _prepare_input_and_forward_pass(subpb, net, knns)
    if sample:
        probs = torch.softmax(scores, dim=-1)
        selected_nodes = torch.tensor(
            [np.random.choice(np.arange(probs.shape[1]), p=prob.detach().cpu().numpy()) for prob in probs],
            device=probs.device
        )[:, None]
    else:
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode="trunc")
    via_depot = (selected_nodes % 2 == 1)
    idx_selected_original = torch.gather(subpb.original_idxs, 1, idx_selected)
    new_subpb, via_depot = _reformat_subproblem_for_next_step(subpb, idx_selected, via_depot)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subpb


# ----------------------------
# Model input preparation
# ----------------------------

def _prepare_input_and_forward_pass(subpb: UniVRPSubPb, net: Module, knns: int) -> Tensor:
    name = subpb.problem_name.lower()
    has_tw = _has_tw(name)
    has_dc = _has_dc(name)

    bs, num_nodes, _, num_features = subpb.dist_matrices.shape

    if 0 < knns < num_nodes:
        knn_indices = torch.topk(subpb.dist_matrices[:, :-1, 0, 0], k=knns - 1, dim=-1, largest=False).indices
        # 追加仓库（末尾）
        knn_indices = torch.cat([knn_indices, torch.full([bs, 1], num_nodes - 1, device=knn_indices.device)], dim=-1)

        knn_node_demands = torch.gather(subpb.node_demands, 1, knn_indices)
        knn_dist = torch.gather(subpb.dist_matrices, 1, knn_indices[..., None, None].repeat(1, 1, num_nodes, num_features))
        knn_dist = torch.gather(knn_dist, 2, knn_indices[:, None, :, None].repeat(1, knns, 1, num_features))
        norm_term = knn_dist.reshape(bs, -1).amax(dim=-1).clamp_min(1e-9)
        knn_edge_features = knn_dist / norm_term[:, None, None, None]

        if has_tw:
            knn_service_times = torch.gather(subpb.service_times, 1, knn_indices)
            tw_dim = subpb.time_windows.shape[-1]
            knn_time_windows = torch.gather(subpb.time_windows, 1, knn_indices.unsqueeze(-1).repeat(1, 1, tw_dim))

            # ---- 传 11 段超集（TW + DC 占位）----
            data = [
                knn_dist,                                      # dist_all
                knn_node_demands,                              # demands
                subpb.remaining_capacities[:, 0][..., None],   # total_cap
                subpb.remaining_capacities[:, -1][..., None],  # remaining_cap
                None,                                          # via_depots
                knn_service_times,                             # service_times
                knn_time_windows,                              # time_windows
                subpb.departure_times[:, -1][..., None],       # departure_times
                None,                                          # distance_constraints (placeholder)
                None,                                          # remaining_distance_constraints (placeholder)
                None,                                          # optimal place-holder
            ]
            node_features, _, problem_data = prepare_routing_data(data, "cvrptw")
            scores_knn = net(node_features, knn_edge_features, problem_data)

            # ====== TW masking for KNN branch ======
            dep = subpb.departure_times[:, -1].to(knn_dist.dtype).to(knn_dist.device)  # (B,)
            travel_kk = knn_dist[..., 0]                 # (B,K,K)
            tw_earliest = knn_time_windows[..., 0]       # (B,K)
            tw_latest   = knn_time_windows[..., 1]       # (B,K)

            eta_direct = dep[:, None] + travel_kk[:, 0, :]     # (B,K)
            eta_direct = torch.maximum(eta_direct, tw_earliest)
            # 约定 KNN 末尾为 depot
            eta_via = travel_kk[:, -1, :]
            eta_via = torch.maximum(eta_via, tw_earliest)

            mask_direct = (eta_direct > tw_latest)      # (B,K)
            mask_via    = (eta_via    > tw_latest)      # (B,K)

            mask_knn = torch.zeros_like(scores_knn, dtype=torch.bool)  # (B,2K)
            mask_knn[:, 0::2] = mask_direct
            mask_knn[:, 1::2] = mask_via
            scores_knn = scores_knn.masked_fill(mask_knn, float("-inf"))
            # ====== end TW masking for KNN branch ======

            # ====== D masking for KNN branch (if also has d) ======
            if has_dc and (subpb.remaining_distances is not None):
                rem_last = subpb.remaining_distances[:, -1]  # (B,)
                rem_init = subpb.remaining_distances[:,  0]  # (B,)
                direct_leg = travel_kk[:, 0, :]              # (B,K) 0->j
                via_leg    = travel_kk[:, 0, :]              # (B,K) 回仓后从 0->j
                mask_d_direct = direct_leg > rem_last[:, None]
                mask_d_via    = via_leg    > rem_init[:, None]

                mask_knn_d = torch.zeros_like(scores_knn, dtype=torch.bool)  # (B,2K)
                mask_knn_d[:, 0::2] = mask_d_direct
                mask_knn_d[:, 1::2] = mask_d_via
                scores_knn = scores_knn.masked_fill(mask_knn_d, float("-inf"))
            # ====== end D masking for KNN branch ======

        else:
            # ---- 非 TW 情况，仍旧传 11 段（TW 占位为 None）----
            if has_dc and (subpb.remaining_distances is not None):
                data = [
                    knn_dist,
                    knn_node_demands,
                    subpb.remaining_capacities[:, 0][..., None],
                    subpb.remaining_capacities[:, -1][..., None],
                    None,   # via_depots
                    None,   # service_times
                    None,   # time_windows
                    None,   # departure_times
                    subpb.remaining_distances[:, 0][..., None],   # distance_constraints
                    subpb.remaining_distances[:, -1][..., None],  # remaining_distance_constraints
                    None,
                ]
            else:
                data = [
                    knn_dist,
                    knn_node_demands,
                    subpb.remaining_capacities[:, 0][..., None],
                    subpb.remaining_capacities[:, -1][..., None],
                    None,   # via_depots
                    None,   # service_times
                    None,   # time_windows
                    None,   # departure_times
                    None,   # distance_constraints
                    None,   # remaining_distance_constraints
                    None,
                ]
            node_features, _, problem_data = prepare_data(data, subpb.problem_name)
            scores_knn = net(node_features, knn_edge_features, problem_data)

        # 将 KNN 分数散射回全长
        scores_full = torch.full((bs, 2 * num_nodes), -np.inf, device=subpb.dist_matrices.device)
        double_knn_indices = torch.zeros([bs, 2 * knns], device=knn_indices.device, dtype=torch.long)
        double_knn_indices[:, 0::2] = 2 * knn_indices
        double_knn_indices[:, 1::2] = 2 * knn_indices + 1
        scores = torch.scatter(scores_full, 1, double_knn_indices, scores_knn)

    else:
        if has_tw:
            # ---- 传 11 段超集（TW + DC 占位）----
            data = [
                subpb.dist_matrices,                            # dist_all
                subpb.node_demands,                             # demands
                subpb.remaining_capacities[:, 0].unsqueeze(-1), # total_cap
                subpb.remaining_capacities[:, -1].unsqueeze(-1),# remaining_cap
                None,                                           # via_depots
                subpb.service_times,                            # service_times
                subpb.time_windows,                             # time_windows
                subpb.departure_times[:, -1].unsqueeze(-1),     # departure_times
                None,                                           # distance_constraints (placeholder)
                None,                                           # remaining_distance_constraints (placeholder)
                None,                                           # optimal placeholder
            ]
            node_features, edge_features, problem_data = prepare_routing_data(data, "cvrptw")
            scores = net(node_features, edge_features, problem_data)

            # ====== TW masking for full-graph branch ======
            dep = subpb.departure_times[:, -1].to(subpb.dist_matrices.dtype).to(subpb.dist_matrices.device)  # (B,)
            travel_nn = subpb.dist_matrices[..., 0]     # (B,N,N)
            tw_earliest = subpb.time_windows[..., 0]    # (B,N)
            tw_latest   = subpb.time_windows[..., 1]    # (B,N)

            eta_direct = dep[:, None] + travel_nn[:, 0, :]  # (B,N)
            eta_direct = torch.maximum(eta_direct, tw_earliest)

            eta_via = travel_nn[:, -1, :]                   # (B,N)
            eta_via = torch.maximum(eta_via, tw_earliest)

            mask_direct = (eta_direct > tw_latest)          # (B,N)
            mask_via    = (eta_via    > tw_latest)          # (B,N)

            scores[:, 0::2] = scores[:, 0::2].masked_fill(mask_direct, float("-inf"))
            scores[:, 1::2] = scores[:, 1::2].masked_fill(mask_via,    float("-inf"))
            # ====== end TW masking for full-graph branch ======

            # ====== D masking for full-graph branch (if also has d) ======
            if has_dc and (subpb.remaining_distances is not None):
                rem_last = subpb.remaining_distances[:, -1]  # (B,)
                rem_init = subpb.remaining_distances[:,  0]  # (B,)
                direct_leg = travel_nn[:, 0, :]               # (B,N)
                via_leg    = travel_nn[:, 0, :]               # (B,N)
                mask_d_direct = direct_leg > rem_last[:, None]
                mask_d_via    = via_leg    > rem_init[:, None]
                scores[:, 0::2] = scores[:, 0::2].masked_fill(mask_d_direct, float("-inf"))
                scores[:, 1::2] = scores[:, 1::2].masked_fill(mask_d_via,    float("-inf"))
            # ====== end D masking for full-graph branch ======

        else:
            # ---- 非 TW 情况，也传 11 段（TW 占位 None）----
            if has_dc and (subpb.remaining_distances is not None):
                data = [
                    subpb.dist_matrices,
                    subpb.node_demands,
                    subpb.remaining_capacities[:, 0].unsqueeze(-1),
                    subpb.remaining_capacities[:, -1].unsqueeze(-1),
                    None,   # via_depots
                    None,   # service_times
                    None,   # time_windows
                    None,   # departure_times
                    subpb.remaining_distances[:, 0].unsqueeze(-1),   # distance_constraints
                    subpb.remaining_distances[:, -1].unsqueeze(-1),  # remaining_distance_constraints
                    None,
                ]
            else:
                data = [
                    subpb.dist_matrices,
                    subpb.node_demands,
                    subpb.remaining_capacities[:, 0].unsqueeze(-1),
                    subpb.remaining_capacities[:, -1].unsqueeze(-1),
                    None,   # via_depots
                    None,   # service_times
                    None,   # time_windows
                    None,   # departure_times
                    None,   # distance_constraints
                    None,   # remaining_distance_constraints
                    None,
                ]
            node_features, edge_features, problem_data = prepare_data(data, subpb.problem_name)
            scores = net(node_features, edge_features, problem_data)

    # 起始步屏蔽 via-depot（更稳健判据：历史栈仅 [total, current]）
    is_first_step = subpb.remaining_capacities.size(1) == 2
    if is_first_step:
        scores[:, 1::2] = -np.inf

    # 统一禁止将“仓库本身（N-1）”作为目标节点（直达/经仓两种动作都禁用）
    depot_even = 2 * (num_nodes - 1)
    depot_odd  = depot_even + 1
    scores[:, depot_even:depot_odd + 1] = -np.inf

    return scores


# ----------------------------
# State transition
# ----------------------------

def _reformat_subproblem_for_next_step(subpb: UniVRPSubPb,
                                       idx_selected: Tensor,
                                       via_depot: Tensor) -> tuple[UniVRPSubPb, Tensor]:
    name = subpb.problem_name.lower()
    has_tw = _has_tw(name)
    has_bc = _has_bc(name)
    has_dc = _has_dc(name)

    bs, subpb_size, _, _ = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    is_selected = (torch.arange(subpb_size, device=device).unsqueeze(0).repeat(bs, 1)
                   == idx_selected.repeat(1, subpb_size))

    # 取出选中节点信息，构造重排后的 2D 张量
    selected_demands = subpb.node_demands[is_selected].unsqueeze(1)  # (B,1)
    next_demands = remove_origin_and_reorder_tensor(subpb.node_demands, is_selected)
    next_original_idxs = remove_origin_and_reorder_tensor(subpb.original_idxs, is_selected)

    # 容量更新（先按“直达”的逻辑预更新，随后根据 via_depot/强制via 调整）
    total_cap = subpb.remaining_capacities[:, 0:1]                 # (B,1)
    prev_cap  = subpb.remaining_capacities[:, -1].unsqueeze(-1)    # (B,1)

    if has_bc:
        cand_cap = prev_cap - selected_demands
        # 暂不更改 via_depot；先记录 cand，后续如 via=True 则回满再应用，并 clamp
    else:
        cand_cap = prev_cap - selected_demands
        # 不足则标记 via
        via_depot = via_depot | (cand_cap < 0)

    # ---- TW：计算直达/经仓的 departure，并在直达违背、经仓可行时强制 via ----
    if has_tw:
        assert (subpb.service_times is not None) and (subpb.time_windows is not None) and (subpb.departure_times is not None)

        selected_service_times = subpb.service_times[is_selected].unsqueeze(1)   # (B,1)
        selected_time_windows = subpb.time_windows[is_selected].unsqueeze(1)     # (B,1,2)

        direct_travel = subpb.dist_matrices[:, 0, :, 0][is_selected].unsqueeze(-1)  # (B,1)
        arrive_direct = subpb.departure_times[:, -1].unsqueeze(-1) + direct_travel
        arrive_direct = torch.maximum(arrive_direct, selected_time_windows[..., 0])
        depart_direct = arrive_direct + selected_service_times

        from_depot = subpb.dist_matrices[:, -1, :, 0][is_selected].unsqueeze(-1)    # (B,1)
        arrive_via  = torch.maximum(from_depot, selected_time_windows[..., 0])
        depart_via  = arrive_via + selected_service_times

        latest_tw = subpb.time_windows[
            torch.arange(bs, device=subpb.time_windows.device), idx_selected.squeeze(1), 1
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

    # ---- 容量最终确定（考虑 via_depot 与 b 族 clamp）----
    if has_bc:
        # via: 回满再应用 & clamp
        next_cap_via  = torch.clamp(total_cap - selected_demands, min=torch.zeros_like(total_cap), max=total_cap)
        next_cap_dir  = torch.clamp(cand_cap, min=torch.zeros_like(total_cap), max=total_cap)
        next_cap = torch.where(via_depot, next_cap_via, next_cap_dir)
    else:
        # 非 b 族：via 已在上面标记；via 时回满再应用
        next_cap_via = total_cap - selected_demands
        next_cap = torch.where(via_depot, next_cap_via, cand_cap)

    next_remaining_capacities = torch.cat([subpb.remaining_capacities, next_cap], dim=-1)

    # ---- 距离约束最终确定 ----
    if has_dc:
        assert subpb.remaining_distances is not None
        direct_leg = subpb.dist_matrices[:, 0, :, 0][is_selected].unsqueeze(-1)  # (B,1) 0->sel
        from_depot_leg = subpb.dist_matrices[:, 0, :, 0][is_selected].unsqueeze(-1)  # (B,1) 0->sel（回仓后再出发）
        rem_prev = subpb.remaining_distances[:, -1].unsqueeze(-1)                 # (B,1)
        rem_init = subpb.remaining_distances[:, 0:1]                              # (B,1)

        rem_dir = rem_prev - direct_leg
        rem_via = rem_init - from_depot_leg
        next_rem_dist = torch.where(via_depot, rem_via, rem_dir)
        next_remaining_distances = torch.cat([subpb.remaining_distances, next_rem_dist], dim=-1)
    else:
        next_remaining_distances = None

    # ---- 重排矩阵到下一步 ----
    next_dist_matrices = remove_origin_and_reorder_matrix(subpb.dist_matrices, is_selected)

    # 2D 的 service_times 可通用重排
    next_service_times = (remove_origin_and_reorder_tensor(subpb.service_times, is_selected)
                          if subpb.service_times is not None else None)

    # 3D 的 time_windows 需要单独重排
    if subpb.time_windows is not None:
        tw_dim = subpb.time_windows.size(-1)  # 通常 2
        selected_tw = subpb.time_windows[is_selected].unsqueeze(1)               # (B,1,tw_dim)
        remaining_tw = subpb.time_windows[~is_selected].reshape(bs, -1, tw_dim)  # (B,N,tw_dim)
        remaining_tw = remaining_tw[:, 1:, :]                                    # 去掉旧首元素
        next_time_windows = torch.cat([selected_tw, remaining_tw], dim=1)
    else:
        next_time_windows = None

    new_subpb = UniVRPSubPb(
        problem_name=subpb.problem_name,
        dist_matrices=next_dist_matrices,
        node_demands=next_demands,
        service_times=next_service_times,
        time_windows=next_time_windows,
        departure_times=next_departure_times,
        remaining_capacities=next_remaining_capacities,
        remaining_distances=next_remaining_distances,
        original_idxs=next_original_idxs,
    )
    return new_subpb, via_depot
