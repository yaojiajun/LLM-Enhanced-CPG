# -*- coding: utf-8 -*-
"""
VRPTW-only decoding (supports 9/11 unified packs)
- 仅容量 + 时间窗约束；无距离上限/DC、无 backhaul；
- 使用 prepare_routing_data 走模型前向；
- 所有遮罩均采用 masked_fill（无原地修改，兼容 InferenceMode）。
"""

from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, List, Tuple

from learning.reformat_subproblems import remove_origin_and_reorder_matrix
from utils.data_manipulation import prepare_routing_data


# ======== 小工具：按“去掉旧起点 + 选中节点置前”的逻辑重排一维/三维张量 ========

def _reorder_1d(t: Optional[Tensor], is_selected: Tensor) -> Optional[Tensor]:
    """t: (B,N) or None; 选中节点成为新 row-0，其余按原顺序、同时移除旧 row-0。"""
    if t is None:
        return None
    B = is_selected.size(0)
    sel = t[is_selected].unsqueeze(1)                 # (B,1)
    rem = t[~is_selected].reshape(B, -1)[:, 1:]       # (B,N-1) 去掉旧首元素
    return torch.cat([sel, rem], dim=1)               # (B,N)

def _reorder_tw(tw: Optional[Tensor], is_selected: Tensor) -> Optional[Tensor]:
    """tw: (B,N,2) or None；与 _reorder_1d 类似，但最后维度保留。"""
    if tw is None:
        return None
    B = is_selected.size(0)
    D = tw.size(-1)
    sel = tw[is_selected].unsqueeze(1)                               # (B,1,2)
    rem = tw[~is_selected].reshape(B, -1, D)[:, 1:, :]               # (B,N-1,2)
    return torch.cat([sel, rem], dim=1)                               # (B,N,2)


# ===================== 子问题状态 =====================

@dataclass
class VRPTWSubPb:
    problem_name: str
    dist_matrices: Tensor           # (B,N,N,2)
    node_demands: Tensor            # (B,N)
    service_times: Tensor           # (B,N)
    time_windows: Tensor            # (B,N,2)
    departure_times: Tensor         # (B,t) 列末为当前出发时间
    remaining_capacities: Tensor    # (B,t) 列0=总容量；列末=当前剩余
    original_idxs: Tensor           # (B,N)

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


# ===================== tour 重建与长度 =====================

def _reconstruct_tours(paths: Tensor, via_depots: Tensor) -> list[list[int]]:
    """将 via_depot 标记展开为显式回仓 0；末尾保留 paths 的最后元素（通常是 N-1 的 depot）。"""
    bs = paths.shape[0]
    tours = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_next = paths[:, pos].tolist()
        via_pos = (via_depots[:, pos]).nonzero().squeeze(-1).cpu().tolist()
        for b in via_pos:
            if pos == 1:               # 第一步的 “via” 无意义，跳过
                continue
            tours[b].append(0)         # 中途回仓
        for b in range(bs):
            tours[b].append(nodes_next[b])
    return tours

def _sum_tour_len_closed(D_all: Tensor, tours: list[list[int]]) -> Tensor:
    """标准 VRPTW：所有段都计入（包含回仓段）。"""
    lens = []
    B = D_all.size(0)
    for b in range(B):
        seq = tours[b]
        D = D_all[b]                                 # (N,N)
        u = torch.tensor(seq[:-1], device=D.device, dtype=torch.long)
        v = torch.tensor(seq[1:],  device=D.device, dtype=torch.long)
        seg = D[u, v]                                # (L-1,)
        lens.append(seg.sum())
    return torch.stack(lens).to(torch.float32)


# ===================== 对外 API =====================

def decode(problem_name: str,
           data: list,
           net: Module,
           beam_size: int = 1,
           knns: int = -1,
           sample: bool = False,
           make_tours: bool = True) -> Tuple[Optional[Tensor], Optional[List[List[int]]]]:
    """
    期望 data 为 9 段或 11 段：
      9 段:  [dist, demand, total_cap, remaining_cap, via(占位), service_times, time_windows, departure_times, optimal(占位)]
     11 段:  [dist, demand, total_cap, remaining_cap, via(占位), service_times, time_windows, departure_times, dc(占位), rdc(占位), optimal(占位)]
    其它字段（dc/rdc/via/optimal）会被忽略。
    """
    L = len(data)
    if L == 11:
        (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
         service_times, time_windows, departure_times0, *_ignored) = data
    elif L == 9:
        (dist_matrices, node_demands, total_capacities, remaining_cap0, _,
         service_times, time_windows, departure_times0, _ignored2) = data
    else:
        raise ValueError(f"VRPTW decoder expects data length in {{9,11}}, got {L}")

    bs, num_nodes, _, _ = dist_matrices.shape

    # 出发时间兜底
    if (departure_times0 is None) or (departure_times0.dim() == 2 and departure_times0.size(1) == 0):
        departure_times = torch.zeros(bs, 1, dtype=dist_matrices.dtype, device=dist_matrices.device)
    else:
        departure_times = departure_times0.to(dist_matrices.dtype).to(dist_matrices.device)

    # 统一 dtype/device
    service_times = service_times.to(dist_matrices.dtype).to(dist_matrices.device)
    time_windows  = time_windows.to(dist_matrices.dtype).to(dist_matrices.device)

    # 容量栈：[总容量, 当前剩余]
    remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)  # (B,2)

    # 原始索引
    original_idxs = torch.arange(num_nodes, device=dist_matrices.device)[None, :].repeat(bs, 1)

    subpb = VRPTWSubPb(
        problem_name=problem_name,
        dist_matrices=dist_matrices,
        node_demands=node_demands,
        service_times=service_times,
        time_windows=time_windows,
        departure_times=departure_times,
        remaining_capacities=remaining_capacities,
        original_idxs=original_idxs,
    )

    if beam_size != 1:
        raise NotImplementedError("VRPTW-only decoder: beam search is not implemented.")

    paths, via_depots, subpb = _greedy_decoding_loop(subpb, net, sample)

    # 合法性：每行必须恰好包含 1..N-2
    sorted_sel = torch.sort(paths[:, 1:-1], dim=1).values
    expect = torch.arange(1, num_nodes - 1, device=paths.device).expand(paths.size(0), -1)
    assert torch.equal(sorted_sel, expect), "paths contain non-customer indices or duplicates"

    # 基本可行性：容量非负、且不超过总容量；时间窗由遮罩与转移逻辑保证
    assert (subpb.remaining_capacities >= 0).all()
    total_cap = subpb.remaining_capacities[:, 0:1]
    assert (subpb.remaining_capacities <= total_cap + 1e-6).all()

    if make_tours:
        tours = _reconstruct_tours(paths, via_depots)
        D_all = dist_matrices[..., 0]  # (B,N,N) 使用通道 0 的行驶时间/距离
        tour_lens = _sum_tour_len_closed(D_all, tours)
    else:
        tours = None
        tour_lens = None

    return tour_lens, tours


# ===================== Greedy 解码循环 =====================

def _greedy_decoding_loop(subpb: VRPTWSubPb, net: Module, sample: bool):
    bs, num_nodes, _, _ = subpb.dist_matrices.shape
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=subpb.dist_matrices.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=subpb.dist_matrices.device)
    paths[:, 0] = 0                 # 起点仓库
    paths[:, -1] = num_nodes - 1    # 末尾 depot（在 VRPTW 中是真实仓库）

    for _ in range(1, num_nodes - 1):
        idx_selected, via_depot, subpb = _greedy_decoding_step(subpb, net, sample)
        # 把选中的“原图索引”落回 path；via 标记也记录
        paths[:, subpb.original_idxs.size(1) - (subpb.dist_matrices.size(1))] = idx_selected
        # 上面这句为了安全起见可改为逐次累加 pos，这里简化为一次性写入：
        # 实际更稳妥的写法是我们维护一个外层循环计数；这里改为简单方案：
        # 重新写一次更直观：
        pass

    # 更直观的实现（避免上面的复杂映射）：
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=subpb.dist_matrices.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=subpb.dist_matrices.device)
    paths[:, 0] = 0
    paths[:, -1] = num_nodes - 1
    sub = subpb
    for dec_pos in range(1, num_nodes - 1):
        idx_selected, via_depot, sub = _greedy_decoding_step(sub, net, sample)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot
    return paths, via_depots, sub


def _greedy_decoding_step(subpb: VRPTWSubPb, net: Module, sample: bool):
    scores = _forward_and_mask(subpb, net)  # (B, 2N)

    if sample:
        probs = torch.softmax(scores, dim=-1)
        sel = torch.tensor(
            [np.random.choice(np.arange(probs.shape[1]), p=prob.detach().cpu().numpy()) for prob in probs],
            device=probs.device
        )[:, None]
    else:
        sel = torch.argmax(scores, dim=1, keepdim=True)

    idx_selected = torch.div(sel, 2, rounding_mode="trunc")   # (B,1) 0..N-1
    via_depot = (sel % 2 == 1)                                # (B,1) True: 经仓

    # 将“原图索引”取回（subpb.original_idxs 的 row-0 是当前 origin，其他是剩余客户与 depot）
    idx_selected_original = torch.gather(subpb.original_idxs, 1, idx_selected)

    new_subpb, via_depot = _step_transition(subpb, idx_selected, via_depot)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subpb


def _forward_and_mask(subpb: VRPTWSubPb, net: Module) -> Tensor:
    bs, num_nodes, _, _ = subpb.dist_matrices.shape

    # ---- 构造输入（11 段超集；dc/rdc 为空）----
    data = [
        subpb.dist_matrices,                               # dist_all
        subpb.node_demands,                                # demands
        subpb.remaining_capacities[:, 0].unsqueeze(-1),    # total_cap
        subpb.remaining_capacities[:, -1].unsqueeze(-1),   # remaining_cap
        None,                                              # via_depots (unused)
        subpb.service_times,                               # service_times
        subpb.time_windows,                                # time_windows
        subpb.departure_times[:, -1].unsqueeze(-1),        # departure_times
        None, None, None,
    ]
    node_features, edge_features, problem_data = prepare_routing_data(data, "cvrptw")
    scores = net(node_features, edge_features, problem_data)        # (B, 2N)

    # ---- 容量遮罩：偶(直达)用当前剩余；奇(经仓)用总容量 ----
    demand  = subpb.node_demands                                   # (B,N)
    cur_cap = subpb.remaining_capacities[:, -1].unsqueeze(-1)      # (B,1)
    tot_cap = subpb.remaining_capacities[:,  0].unsqueeze(-1)      # (B,1)

    mask_direct = demand > cur_cap
    mask_via    = demand > tot_cap
    mask_2n = torch.zeros_like(scores, dtype=torch.bool)           # (B,2N)
    mask_2n[:, 0::2] = mask_direct
    mask_2n[:, 1::2] = mask_via
    scores = scores.masked_fill(mask_2n, float("-inf"))

    # ---- 时间窗遮罩：直达用 row-0 → j；经仓用 depot(-1) → j ----
    travel_nn  = subpb.dist_matrices[..., 0]         # (B,N,N)
    dep        = subpb.departure_times[:, -1]        # (B,)
    tw_earliest = subpb.time_windows[..., 0]         # (B,N)
    tw_latest   = subpb.time_windows[..., 1]         # (B,N)

    eta_direct = dep[:, None] + travel_nn[:, 0, :]   # (B,N)
    eta_direct = torch.maximum(eta_direct, tw_earliest)
    eta_via    = travel_nn[:, -1, :]                 # (B,N)  depot -> j
    eta_via    = torch.maximum(eta_via, tw_earliest)

    tw_mask = torch.zeros_like(scores, dtype=torch.bool)
    tw_mask[:, 0::2] = (eta_direct > tw_latest)
    tw_mask[:, 1::2] = (eta_via    > tw_latest)
    scores = scores.masked_fill(tw_mask, float("-inf"))

    # 起始步禁止“经仓”（历史栈仅 [total,current] → 长度==2）
    is_first_step = subpb.remaining_capacities.size(1) == 2
    if is_first_step:
        mask_first = torch.zeros_like(scores, dtype=torch.bool)
        mask_first[:, 1::2] = True
        scores = scores.masked_fill(mask_first, float("-inf"))

    # 禁止把“仓库本身”（N-1）当成目标（直达/经仓皆禁）
    depot_even = 2 * (num_nodes - 1)
    depot_odd  = depot_even + 1
    mask_depot = torch.zeros_like(scores, dtype=torch.bool)
    mask_depot[:, depot_even:depot_odd + 1] = True
    scores = scores.masked_fill(mask_depot, float("-inf"))

    return scores


def _step_transition(subpb: VRPTWSubPb,
                     idx_selected: Tensor,
                     via_depot: Tensor) -> tuple[VRPTWSubPb, Tensor]:
    """
    - 计算直达/经仓到达时间并带服务时间推进出发时刻；
    - 可行时保留选择；若直达违背而“经仓可行”，则强制 via；
    - 容量：via 表示先回满再去客户。
    """
    bs, subpb_size, _, _ = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    is_selected = (torch.arange(subpb_size, device=device).unsqueeze(0).repeat(bs, 1)
                   == idx_selected.repeat(1, subpb_size))  # (B,N)

    # --- 需求 / 时间窗 / 服务时间（选中值）---
    selected_demand = subpb.node_demands[is_selected].unsqueeze(1)         # (B,1)
    selected_st     = subpb.service_times[is_selected].unsqueeze(1)        # (B,1)
    selected_tw     = subpb.time_windows[is_selected].unsqueeze(1)         # (B,1,2)

    # --- 出发与行驶时间 ---
    travel_nn  = subpb.dist_matrices[..., 0]                                # (B,N,N)
    dep_prev   = subpb.departure_times[:, -1].unsqueeze(-1)                 # (B,1)

    direct_leg = travel_nn[:, 0, :][is_selected].unsqueeze(-1)              # (B,1) 从当前 origin(0) 直达
    via_leg    = travel_nn[:, -1, :][is_selected].unsqueeze(-1)             # (B,1) 从 depot 直达

    arrive_direct = torch.maximum(dep_prev + direct_leg, selected_tw[..., 0])
    depart_direct = arrive_direct + selected_st

    arrive_via = torch.maximum(via_leg, selected_tw[..., 0])                # 回仓后从 depot 出发
    depart_via = arrive_via + selected_st

    latest_tw = selected_tw[..., 1]                                         # (B,1)

    direct_viol = (arrive_direct > latest_tw).squeeze(-1)
    via_viol    = (arrive_via    > latest_tw).squeeze(-1)
    force_via   = direct_viol & (~via_viol)                                  # 直达违背、经仓可行 → 强制 via
    if force_via.any():
        via_depot[force_via] = True

    next_depart = torch.where(via_depot, depart_via, depart_direct)          # (B,1)
    next_departure_times = torch.cat([subpb.departure_times, next_depart], dim=-1)

    # --- 容量：不足则强制 via；via 表示回满再去客户 ---
    total_cap = subpb.remaining_capacities[:, 0:1]                           # (B,1)
    prev_cap  = subpb.remaining_capacities[:, -1].unsqueeze(-1)              # (B,1)
    cand_cap  = prev_cap - selected_demand                                   # (B,1)
    via_depot = via_depot | (cand_cap < 0)

    next_cap_via = total_cap - selected_demand                               # (B,1)
    next_cap     = torch.where(via_depot, next_cap_via, cand_cap)
    next_remaining_capacities = torch.cat([subpb.remaining_capacities, next_cap], dim=-1)

    # --- 子图重排（去掉旧 origin，并把“选中节点”置为新的 row-0）---
    next_dist_matrices = remove_origin_and_reorder_matrix(subpb.dist_matrices, is_selected)
    next_demands       = _reorder_1d(subpb.node_demands, is_selected)
    next_service_times = _reorder_1d(subpb.service_times, is_selected)
    next_time_windows  = _reorder_tw(subpb.time_windows, is_selected)
    next_original_idxs = _reorder_1d(subpb.original_idxs, is_selected)

    new_subpb = VRPTWSubPb(
        problem_name=subpb.problem_name,
        dist_matrices=next_dist_matrices,
        node_demands=next_demands,
        service_times=next_service_times,
        time_windows=next_time_windows,
        departure_times=next_departure_times,
        remaining_capacities=next_remaining_capacities,
        original_idxs=next_original_idxs,
    )
    return new_subpb, via_depot
