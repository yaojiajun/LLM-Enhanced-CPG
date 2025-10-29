# -*- coding: utf-8 -*-
"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.nn.modules import ModuleList

from model.adapters import NodeAdapter, EdgeAdapter, OutputAdapter
from model.layer import Layer


class GOAL(Module):
    def __init__(
        self,
        dim_node_idx,
        dim_emb,
        num_layers,
        dim_ff,
        activation_ff,
        node_feature_low_dim,
        edge_feature_low_dim,
        activation_edge_adapter,
        num_heads,
        is_finetuning: bool = False,
    ):
        super().__init__()
        self.dim_node_idx = dim_node_idx

        self.node_idx_projection = Linear(dim_node_idx, dim_emb)
        self.is_finetuning = is_finetuning

        self.nb_layers = num_layers
        self.node_adapter = NodeAdapter(dim_emb, node_feature_low_dim, is_finetuning)
        self.edge_adapter = EdgeAdapter(dim_emb, activation_edge_adapter, edge_feature_low_dim, is_finetuning)
        self.output_adapter = OutputAdapter(dim_emb, is_finetuning)

        self.layers = ModuleList([Layer(dim_emb, dim_ff, num_heads, activation_ff) for _ in range(num_layers)])

    def get_device(self):
        return list(self.state_dict().items())[0][1].device

    def forward(self, node_features, edge_features, problem_data):
        """
        Forward during training/eval (features已按需重排)

        node features
            TSP     None
            CVRP    [B, S, (demand, current_capacity)]
            CVRP-TW [B, S, (demand, service_time, beg_tw, end_tw, departure_times, remaining_capacity)]
            OP      [B, S, (node_value, upper_bound)]
            KP      [B, S, (weight, value, remaining_capacity)]
            MWVC    [B, S, (node_weight)]
            JSP     [[B, S, (job_feature)], [B, S, (machine_feature, machine_state)]]
        """
        batch_size, seq_len, device = self.data_info(node_features, edge_features, problem_data)

        # 可选图问题的mask（MVC/MIS/MCLP）
        mask = self.create_mask(problem_data, edge_features)

        # node随机索引投影（保持原语义）
        node_random_emb = self.node_idx_projection(torch.rand((batch_size, seq_len, self.dim_node_idx), device=device))

        # 输入适配
        state = self.node_adapter(node_features, node_random_emb, problem_data)
        edge_emb = self.edge_adapter(edge_features, problem_data)

        # 主干
        for layer in self.layers:
            state = layer(state, edge_emb, mask, problem_data["is_multitype"], problem_data["seq_len_per_type"])

        # 输出适配
        scores = self.output_adapter(state, problem_data)

        # 不可行动作屏蔽（向量化）
        scores = self.mask_infeasible_actions(scores, mask, problem_data)

        # 展平成 (B, -1)
        return scores.reshape(scores.shape[0], -1)

    @staticmethod
    def mask_infeasible_actions(scores, mask, problem_data):
        """
        高效屏蔽：
        - 统一使用 masked_fill_/fill_，避免二次索引产生临时张量
        - DCVRP 支持快路径：若 problem_data 提供 dist_to_depot / dist_from_depot，则避免 4D 切片
        """
        neg_inf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)

        # ---- 统一小写名
        name = problem_data["problem_name"].lower()

        if name == "tsp":
            scores[:, 0].fill_(neg_inf)
            scores[:, -1].fill_(neg_inf)

        elif name == "trp":
            scores[:, 0].fill_(neg_inf)

        elif name == "sop":
            scores[:, 0].fill_(neg_inf)
            # 任意存在先序约束则整行屏蔽
            has_prec = (problem_data["dist_matrices"][..., 0:1].squeeze(-1)[..., 1:] < 0).any(dim=-1)
            scores[has_prec] = neg_inf

        elif name == "pctsp":
            scores[:, 0].fill_(neg_inf)
            scores[:, -1].masked_fill_(problem_data["remaining_to_collect"] > 0, neg_inf)

        # ===== 1) CVRP族（容量屏蔽，含 o/b/d/tw 组合；TW 和 D 也会在后两段叠加各自规则）
        CAP_GROUP = {
            # 无TW
            "cvrp", "ocvrp", "bcvrp", "obcvrp",
            "dcvrp", "odcvrp", "bdcvrp", "obdcvrp",
            # 有TW
            "cvrptw", "ocvrptw", "bcvrptw", "obcvrptw",
            "dcvrptw", "odcvrptw", "bdcvrptw", "obdcvrptw",
        }
        if name in CAP_GROUP:
            # 起点/终点列屏蔽（direct/via 两列）
            scores[:, 0, :].fill_(neg_inf)
            scores[:, -1, :].fill_(neg_inf)

            # 剩余容量 (B, 1)
            rem_cap = problem_data["remaining_capacities"].unsqueeze(-1)
            rem_backhaul_cap = problem_data["remaining_backhaul_capacities"].unsqueeze(-1)
            # 全量需求 (B, S)
            node_demands = problem_data["node_demands"]

            # 拆分：正值 -> dem（送货），负值 -> dem_backhaul（取货/回程）
            # dem ≥ 0，dem_backhaul ≤ 0
            dem = torch.clamp(node_demands, min=0)
            dem_backhaul = torch.clamp(node_demands, max=0)

            # 送货容量校验：直接与 rem_cap 对比
            delta = rem_cap - dem  # (B, S)
            direct_violate = (delta < 0)
            delta1 = rem_backhaul_cap+dem_backhaul
            direct_violate1 = (delta1 < 0 )

            # 仅对 direct 列应用：[..., 0] 是 direct，[..., 1] 是 via
            scores[..., 0].masked_fill_(direct_violate, neg_inf)
            scores[..., 0].masked_fill_(direct_violate1, neg_inf)
        # ===== 2) DCVRP族（距离屏蔽：凡含 d 的都加入；会与上面容量屏蔽叠加）
        DIST_GROUP = {
            "dcvrp", "odcvrp", "bdcvrp", "obdcvrp",
            "dcvrptw", "odcvrptw", "bdcvrptw", "obdcvrptw",
        }
        if name in DIST_GROUP:
            B, S, _ = scores.shape
            mid_len = S - 2
            eps = 1e-6

            # 走/回仓距离（优先用预计算）
            d2d = problem_data.get("dist_to_depot", None)     # (B, S)
            df0 = problem_data.get("dist_from_depot", None)   # (B, S)
            if (d2d is not None) and (df0 is not None):
                dist_go   = d2d[:, 1:1 + mid_len]    # depot->node
                dist_back = df0[:, 1:1 + mid_len]    # node->depot
            else:
                dm = problem_data["dist_matrices"]           # (B, N, N, 2)
                dist_go   = dm[:, 0, 1:1 + mid_len, 0]
                dist_back = dm[:, 1:1 + mid_len, -1, 0]

            # 剩余可用距离
            rdc = problem_data["remaining_distance_constraints"]
            if rdc.dim() == 1:
                rdc = rdc[:, None]
            if rdc.dim() == 2 and rdc.size(1) != mid_len:
                if rdc.size(1) >= 1 + mid_len:
                    rdc = rdc[:, 1:1 + mid_len]
                elif rdc.size(1) == 1:
                    rdc = rdc.expand(-1, mid_len)
                else:
                    last = rdc[:, -1:]
                    rdc  = torch.cat([rdc, last.expand(-1, max(0, mid_len - rdc.size(1)))], dim=1)
            elif rdc.dim() == 2 and rdc.size(1) == 1:
                rdc = rdc.expand(-1, mid_len)

            over = (dist_go + dist_back - rdc) > eps
            scores[:, 1:1 + mid_len, 0].masked_fill_(over, neg_inf)

        # ===== 3) CVRPTW族（时间窗屏蔽：凡含 tw 的都加入；若也含 d，上面的距离分支已叠加）
        TW_GROUP = {
            "cvrptw", "ocvrptw", "bcvrptw", "obcvrptw",
            "dcvrptw", "odcvrptw", "bdcvrptw", "obdcvrptw",
        }
        if name in TW_GROUP:
            dep = problem_data["departure_times"].unsqueeze(-1)   # (B,1)
            t_go = problem_data["travel_times"][:, 0, :, 0]       # (B, S)
            tw   = problem_data["time_windows"]                   # (B, S, 2)
            arrive = torch.max(dep + t_go, tw[..., 0])            # (B, S)
            late   = tw[..., 1]                                   # (B, S)
            tw_violate = arrive > late
            scores[..., 0].masked_fill_(tw_violate, neg_inf)
            # print(arrive)
        elif name == "op":
            scores[:, 0].fill_(neg_inf)
            scores[:, -1].fill_(neg_inf)
            d = problem_data["dist_matrices"]                             # (B, N, N, 2)
            ub = problem_data["upper_bounds"].unsqueeze(-1)               # (B, 1)
            # 去/回仓总距离超出上界则屏蔽
            mask_op = d[:, 0, :, 0] + d[:, :, -1, 0] - ub > 0             # (B, N)
            scores.masked_fill_(mask_op, neg_inf)

        elif name in ["mvc", "mis", "mclp"]:
            if mask is not None:
                to_use = mask[..., 0] if mask.dim() == 3 and mask.shape[-1] == 1 else mask
                scores.masked_fill_(to_use, neg_inf)

        elif name == "kp":
            filt = problem_data["weights"] > problem_data["remaining_capacities"].unsqueeze(-1)
            scores.masked_fill_(filt, neg_inf)

        elif name == "multikp":
            filt = (problem_data["weights"][:, None, :]
                    > problem_data["remaining_capacities"][..., None])
            scores.masked_fill_(filt, neg_inf)

        elif name == "jssp":
            tasks_with_precedences = problem_data["precedencies"].sum(dim=-1) > 0
            scores.masked_fill_(tasks_with_precedences, neg_inf)

        return scores  # 维持原形状，由 forward 统一 reshape

    @staticmethod
    def data_info(node_features, edge_features, problem_data):
        if problem_data["problem_name"] in ["upms", "jssp", "ossp"]:
            if problem_data["problem_name"] in ["jssp", "ossp"]:
                edge_features = edge_features[1]
            batch_size = edge_features.shape[0]
            device = edge_features.device
            seq_len = edge_features.shape[1] + edge_features.shape[2]
        elif problem_data["problem_name"] == "multikp":
            batch_size = node_features[0].shape[0]
            device = node_features[0].device
            seq_len = node_features[0].shape[1] + node_features[1].shape[1]
        else:
            features = edge_features if edge_features is not None else node_features
            batch_size = features.shape[0]
            device = features.device
            seq_len = features.shape[1]
        return batch_size, seq_len, device

    @staticmethod
    def create_mask(problem_data, matrices):
        """
        图类问题的可选掩码（MVC/MIS/MCLP）向量化实现：
        - 避免 Python for 循环
        - 直接返回 bool 型张量，后续用 masked_fill_
        """
        if problem_data["problem_name"] == "mvc":
            m = matrices.squeeze(-1)                          # (B, S, S)
            mask = torch.full_like(m, False, dtype=torch.bool)
            mask[m.sum(dim=-1) == 0] = True

        elif problem_data["problem_name"] == "mis":
            mask = torch.full_like(matrices.squeeze(-1), False, dtype=torch.bool)
            mask[~problem_data["can_be_selected"]] = True

        elif problem_data["problem_name"] == "mclp":
            m = matrices[..., 0]                              # (B, S)
            mask = torch.full_like(m, False, dtype=torch.bool)
            already = problem_data.get("already_selected", None)
            if already is not None:
                mask |= already.to(dtype=torch.bool)

        else:
            mask = None

        return mask
