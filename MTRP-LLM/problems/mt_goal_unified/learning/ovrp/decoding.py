# -*- coding: utf-8 -*-
"""
OVRP-only decoding (supports 6/8/9/11 unified packs)
- 仅容量约束；无 TW / DC / BC / 启发式；
- 评分时用 masked_fill，避免 InferenceMode 下的原地修改；
- 总路径长度严格参照 OVRPEnv._get_travel_distance：
  只累计“下一站 != 0(仓库) 且 != N-1(虚拟仓库)”的段。
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
from utils.data_manipulation import prepare_data


def _is_open(name: str) -> bool:
    return "o" in name.lower()  # 仅用于断言/兼容


@dataclass
class OVRPSubPb:
    problem_name: str
    dist_matrices: Tensor           # (B,N,N,2)
    node_demands: Tensor            # (B,N)
    remaining_capacities: Tensor    # (B,t)  列0=总容量；最后一列=当前剩余
    original_idxs: Tensor           # (B,N)

    def dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def reconstruct_tours(paths: Tensor, via_depots: Tensor) -> list[list[int]]:
    """将 via_depot 标记展开为序列中插入 0（回仓）；末尾保留 paths 的最后元素（可能是 N-1）"""
    bs = paths.shape[0]
    tours = [[0] for _ in range(bs)]
    for pos in range(1, paths.shape[1]):
        nodes_next = paths[:, pos].tolist()
        via_pos = (via_depots[:, pos]).nonzero().squeeze(-1).cpu().tolist()
        for b in via_pos:
            if pos == 1:
                continue
            tours[b].append(0)  # 中途回仓
        for b in range(bs):
            tours[b].append(nodes_next[b])
    return tours


def decode(problem_name: str,
           data: list,
           net: Module,
           beam_size: int = 1,
           knns: int = -1,
           sample: bool = False,
           make_tours: bool = True) -> Tuple[Optional[Tensor], Optional[List[List[int]]]]:

    assert _is_open(problem_name), "This file is OVRP-only."

    L = len(data)
    # 统一读取前 4 段：dist_matrices, node_demands, total_capacities, remaining_cap0
    if L in (6, 8, 9, 11):
        dist_matrices = data[0]
        node_demands = data[1]
        total_capacities = data[2]
        remaining_cap0 = data[3]
    else:
        raise ValueError(f"OVRP-only decoder expects data length in {{6,8,9,11}}, got {L}")

    remaining_capacities = torch.cat([total_capacities, remaining_cap0], dim=-1)  # (B,2) -> [total, current]

    bs, num_nodes, _, _ = dist_matrices.shape
    original_idxs = torch.arange(num_nodes, device=dist_matrices.device)[None, :].repeat(bs, 1)

    subpb = OVRPSubPb(
        problem_name=problem_name,
        dist_matrices=dist_matrices,
        node_demands=node_demands,
        remaining_capacities=remaining_capacities,
        original_idxs=original_idxs,
    )

    if beam_size != 1:
        raise NotImplementedError("OVRP-only decoder: beam search is not implemented.")

    paths, via_depots, subpb = _greedy_decoding_loop(subpb, net, sample)

    # 合法性：每行必须恰好包含 1..N-2
    sorted_sel = torch.sort(paths[:, 1:-1], dim=1).values
    expect = torch.arange(1, num_nodes - 1, device=paths.device).expand(paths.size(0), -1)
    assert torch.equal(sorted_sel, expect), "paths contain non-customer indices or duplicates"

    # 基本可行性：容量非负 & 不超过总容量
    assert (subpb.remaining_capacities >= 0).all()
    total_cap = subpb.remaining_capacities[:, 0:1]
    assert (subpb.remaining_capacities <= total_cap + 1e-6).all()

    # ------- 总路长：严格按 OVRPEnv._get_travel_distance 的思想 -------
    if make_tours:
        seqs = reconstruct_tours(paths, via_depots)            # list of list(int)
        D_all = dist_matrices[..., 0]                          # (B,N,N)
        lens = []
        for b in range(bs):
            seq = seqs[b]
            D = D_all[b]
            u = torch.tensor(seq[:-1], device=D.device, dtype=torch.long)
            v = torch.tensor(seq[1:],  device=D.device, dtype=torch.long)
            seg = D[u, v]                                      # (L-1,)
            # 只要“下一站是仓库”就不计该段；N-1 视作“虚拟仓库”
            n_minus_1 = D.size(0) - 1
            not_to_depot = (v != 0) & (v != n_minus_1)
            seg = seg * not_to_depot.float()
            lens.append(seg.sum())
        tour_lens = torch.stack(lens).to(torch.float32)
        tours = seqs
    else:
        tour_lens, tours = None, None

    return tour_lens, tours


def _greedy_decoding_loop(subpb: OVRPSubPb, net: Module, sample: bool):
    bs, num_nodes, _, _ = subpb.dist_matrices.shape
    paths = torch.zeros((bs, num_nodes), dtype=torch.long, device=subpb.dist_matrices.device)
    via_depots = torch.full((bs, num_nodes), False, dtype=torch.bool, device=subpb.dist_matrices.device)
    paths[:, 0] = 0                 # 起点仓库
    paths[:, -1] = num_nodes - 1    # 末尾虚拟仓库（仅占位，最终计算当作“仓库”屏蔽）

    for dec_pos in range(1, num_nodes - 1):  # 依次写入第 1..N-2 个客户
        idx_selected, via_depot, subpb = _greedy_decoding_step(subpb, net, sample)
        paths[:, dec_pos] = idx_selected
        via_depots[:, dec_pos] = via_depot

    return paths, via_depots, subpb


def _greedy_decoding_step(subpb: OVRPSubPb, net: Module, sample: bool):
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

    idx_selected_original = torch.gather(subpb.original_idxs, 1, idx_selected)
    new_subpb, via_depot = _step_transition(subpb, idx_selected, via_depot)
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), new_subpb


def _forward_and_mask(subpb: OVRPSubPb, net: Module) -> Tensor:
    bs, num_nodes, _, _ = subpb.dist_matrices.shape

    # 用 prepare_data 的 11 段超集接口（非 TW/DC 全传 None）
    data = [
        subpb.dist_matrices,                               # dist_all
        subpb.node_demands,                                # demands
        subpb.remaining_capacities[:, 0].unsqueeze(-1),    # total_cap
        subpb.remaining_capacities[:, -1].unsqueeze(-1),   # remaining_cap
        None, None, None, None,                            # via_depots, service_times, time_windows, departure_times
        None, None, None,                                  # distance_constraints, remaining_distance_constraints, optimal
    ]
    node_features, edge_features, problem_data = prepare_data(data, subpb.problem_name)
    scores = net(node_features, edge_features, problem_data)        # (B, 2N)

    # 容量可行性：直达(偶) 用当前剩余；经仓(奇) 用总容量
    demand = subpb.node_demands                                     # (B,N)
    cur_cap = subpb.remaining_capacities[:, -1].unsqueeze(-1)       # (B,1)
    tot_cap = subpb.remaining_capacities[:,  0].unsqueeze(-1)       # (B,1)

    mask_direct = demand > cur_cap                                   # (B,N)
    mask_via    = demand > tot_cap                                   # (B,N)
    mask_2n = torch.zeros_like(scores, dtype=torch.bool)             # (B,2N)
    mask_2n[:, 0::2] = mask_direct
    mask_2n[:, 1::2] = mask_via
    scores = scores.masked_fill(mask_2n, float("-inf"))

    # 起始步禁止“经仓”（历史仅 [total,current] → 长度==2）
    is_first_step = subpb.remaining_capacities.size(1) == 2
    if is_first_step:
        mask_first = torch.zeros_like(scores, dtype=torch.bool)
        mask_first[:, 1::2] = True
        scores = scores.masked_fill(mask_first, float("-inf"))

    # 禁止将“仓库本身”（N-1）作为目标（直达/经仓皆禁）
    depot_even = 2 * (num_nodes - 1)
    depot_odd  = depot_even + 1
    mask_depot = torch.zeros_like(scores, dtype=torch.bool)
    mask_depot[:, depot_even:depot_odd + 1] = True
    scores = scores.masked_fill(mask_depot, float("-inf"))

    return scores


def _step_transition(subpb: OVRPSubPb,
                     idx_selected: Tensor,
                     via_depot: Tensor) -> tuple[OVRPSubPb, Tensor]:

    bs, subpb_size, _, _ = subpb.dist_matrices.shape
    device = subpb.dist_matrices.device

    is_selected = (torch.arange(subpb_size, device=device).unsqueeze(0).repeat(bs, 1)
                   == idx_selected.repeat(1, subpb_size))  # (B,N)

    # 需求与 id 重排
    selected_demands = subpb.node_demands[is_selected].unsqueeze(1)       # (B,1)
    next_demands = remove_origin_and_reorder_tensor(subpb.node_demands, is_selected)
    next_original_idxs = remove_origin_and_reorder_tensor(subpb.original_idxs, is_selected)

    # 容量更新：不足则强制 via；via 表示回满后再去该客户
    total_cap = subpb.remaining_capacities[:, 0:1]                        # (B,1)
    prev_cap  = subpb.remaining_capacities[:, -1].unsqueeze(-1)           # (B,1)
    cand_cap  = prev_cap - selected_demands                                # (B,1)
    via_depot = via_depot | (cand_cap < 0)

    next_cap_via = total_cap - selected_demands                            # (B,1)
    next_cap = torch.where(via_depot, next_cap_via, cand_cap)              # (B,1)
    next_remaining_capacities = torch.cat([subpb.remaining_capacities, next_cap], dim=-1)

    # 子图重排
    next_dist_matrices = remove_origin_and_reorder_matrix(subpb.dist_matrices, is_selected)

    new_subpb = OVRPSubPb(
        problem_name=subpb.problem_name,
        dist_matrices=next_dist_matrices,
        node_demands=next_demands,
        remaining_capacities=next_remaining_capacities,
        original_idxs=next_original_idxs,
    )
    return new_subpb, via_depot
