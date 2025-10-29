# -*- coding: utf-8 -*-
"""Unified VRP dataset generator (CVRP / OVRP / VRPB / VRPL / VRPTW and combos)

含 L 的问题（VRPL/OVRPL/VRPBL/VRPLTW/VRPBLTW/…）按 DCVRP 字段生成：
- dist_matrices: (B, N, N, 2)   # forward + transpose
- distance_constraints: (B, 1)  # 0.2 + 2 * max(pairwise_dist)
- remaining_distances: (B, N)   # repeat(distance_constraints, N)

其它问题不生成 dist_matrices。
TW：service_times (B, N), time_windows (B, N, 2)  # 两端为仓库
"""
import os
import argparse
import numpy as np

# ----------------------------
# knobs
# ----------------------------
BACKHAUL_RATIO = 0.2      # 最后 20% 客户置为负；若不要 backhaul，设为 0.0
CAPACITY_VALUE = 50.0
ROUTE_LEN_LIMIT = 3.0     # 仅语义字段，可留作训练用；模型主要看 distance_* 两项
DEPOT_START = 0.0
DEPOT_END = 4.6
EPS = 1e-8

# ----------------------------
# base utils
# ----------------------------
def generate_locations(problem_size, num_instances, repeat_origin=True):
    """coords: (B, N, 2)。若 repeat_origin=True，末尾追加首点（N -> N+1），形成 [depot, customers..., depot]。"""
    coords = np.random.rand(num_instances, problem_size, 2).astype(np.float32)
    if repeat_origin:
        coords = np.concatenate([coords, coords[:, 0:1, :]], axis=1)
    return coords  # (B, N, 2)

def pairwise_dm(coords):
    """coords: (B, N, 2) -> dist_mats: (B, N, N)（欧氏距离），仅用于计算 max，不保存"""
    B, N, _ = coords.shape
    dm = np.empty((B, N, N), dtype=np.float32)
    for b in range(B):
        c = coords[b]                                  # (N,2)
        diff = c[:, None, :] - c[None, :, :]           # (N,N,2)
        dm[b] = np.sqrt((diff**2).sum(-1))
    return dm


def _common_vrp_fields(coords, problem_size):
    """构造基础字段（不含 dist_matrices）："""
    B, N, _ = coords.shape
    P = N - 2

    node_demands = np.random.randint(1, 10, size=(B, P)).astype(np.float32)
    node_demands = np.concatenate(
        [np.zeros((B, 1), dtype=np.float32),
         node_demands,
         np.zeros((B, 1), dtype=np.float32)],
        axis=1
    )  # (B, N)

    total_capacities = np.full((B,), CAPACITY_VALUE, dtype=np.float32)
    remaining_capacities = np.repeat(total_capacities[:, None], N, axis=1).astype(np.float32)
    via_depots = np.zeros((B, N), dtype=np.int32)
    tour_lens = np.full((B,), 10.0, dtype=np.float32)

    return {
        "node_coords": coords.astype(np.float32),
        "node_demands": node_demands,
        "total_capacities": total_capacities,
        "remaining_capacities": remaining_capacities,
        "via_depots": via_depots,
        "tour_lens": tour_lens,
    }

# ----------------------------
# TW fields (B, N) + (B, N, 2) with depots
# ----------------------------
def _tw_fields_full(coords):
    """service_times_full: (B,N); time_windows_full: (B,N,2)=[early,late]，两端仓库填 [0,4.6] / 0。"""
    B, N, _ = coords.shape
    P = N - 2
    depot = coords[:, :1, :]              # (B,1,2)
    custs = coords[:, 1:-1, :]            # (B,P,2)

    # 1) 客户的 service_times 与 tw_length
    service_times_c = 0.15 + 0.05 * np.random.rand(B, P).astype(np.float32)  # U(0.15, 0.20)
    tw_length_c     = 5 + 0.05 * np.random.rand(B, P).astype(np.float32)  # U(0.15, 0.20)

    # 2) d0i
    d0i = np.linalg.norm(custs - depot, axis=-1).astype(np.float32)          # (B,P)
    d0i_safe = d0i + EPS

    # 3) ei ~ U(1, A-1)，A = (4.6 - service_times - tw_length)/d0i
    A = (DEPOT_END - service_times_c - tw_length_c) / d0i_safe               # (B,P)
    e_scale = A - 2.0
    ei = np.random.rand(B, P).astype(np.float32) * e_scale + 1.0             # (B,P)

    # 4) early/late + 合法化
    early_c = ei * d0i
    late_c  = early_c + tw_length_c
    early_c = np.clip(early_c, DEPOT_START, DEPOT_END)
    late_c  = np.maximum(late_c, early_c)
    late_c  = np.clip(late_c,  DEPOT_START, DEPOT_END)

    # 5) 拼接仓库
    service_times_full = np.zeros((B, N), dtype=np.float32)
    service_times_full[:, 1:-1] = service_times_c

    time_windows_full = np.zeros((B, N, 2), dtype=np.float32)
    time_windows_full[:, 0, 0]  = DEPOT_START
    time_windows_full[:, 0, 1]  = DEPOT_END
    time_windows_full[:, -1, 0] = DEPOT_START
    time_windows_full[:, -1, 1] = DEPOT_END
    time_windows_full[:, 1:-1, 0] = early_c
    time_windows_full[:, 1:-1, 1] = late_c

    return service_times_full, time_windows_full

# ----------------------------
# name parsing
# ----------------------------
def _parse_flags(name: str):
    """
    open: 以 'OVRP' 开头
    B:    包含 'B'
    L:    包含 'L'
    TW:   包含 'TW'
    """
    s = name.upper().strip()
    if s == "CVRP": return dict(open=False, B=False, L=False, TW=False)
    if s == "OVRP": return dict(open=True,  B=False, L=False, TW=False)

    open_ = s.startswith("OVRP")
    core = s.replace("OVRP", "VRP") if open_ else s
    return dict(open=open_, B=("B" in core), L=("L" in core), TW=("TW" in core))

def list_all_canonical():
    bases = ["CVRP", "OVRP"]
    suffixes = ["", "B", "L", "TW", "BL", "BTW", "LTW", "BLTW"]
    names = []
    for base in bases:
        if base == "CVRP":
            names.append("CVRP")
            for sfx in suffixes[1:]:
                names.append("VRP" + sfx)
        else:
            names.append("OVRP")
            for sfx in suffixes[1:]:
                names.append("OVRP" + sfx)
    return names

# ----------------------------
# main generator
# ----------------------------
def generate_instances(problem, problem_size, num_instances=1):
    """
    - L 分支参考 DCVRP：生成 dist_matrices(2ch) + distance_constraints + remaining_distances
    - TW 分支：service_times(B,N) + time_windows(B,N,2)
    - B 分支：最后 k 个客户需求置负
    """
    flags = _parse_flags(problem)
    coords = generate_locations(problem_size + 1, num_instances)  # (B,N,2)
    data = _common_vrp_fields(coords, problem_size)

    # B: backhaul
    if flags["B"]:
        k = int(BACKHAUL_RATIO * problem_size)
        if k > 0:
            start = problem_size - k + 1      # 客户区间 [1..problem_size]
            end   = problem_size + 1          # 右开，不含最后 depot
            data["node_demands"][:, start:end] = -data["node_demands"][:, start:end]

    # L: 按 DCVRP 生成约束 + dist_matrices(2 通道)
    if flags["L"]:
        Bsz, N, _ = data["node_coords"].shape

        # 语义字段：route_length_limit（可选用）
        data["route_length_limit"] = np.full((num_instances, 1), ROUTE_LEN_LIMIT, dtype=np.float32)

        # 仅用于算 max 距离，不保存 dist 矩阵
        dm = pairwise_dm(data["node_coords"])  # (B,N,N)
        max_d = dm.reshape(Bsz, -1).max(axis=1).astype(np.float32)  # (B,)

        # DCVRP 风格的两项，用于让适配器拼出 3 维节点特征（x, y, 约束/剩余）
        distance_constraints = (0.2 + 2.0 * max_d)[:, None]  # (B,1)
        remaining_distances = np.repeat(distance_constraints, N, axis=1)  # (B,N)

        data["distance_constraints"] = distance_constraints
        data["remaining_distances"] = remaining_distances

    # TW: service_times + time_windows（含仓库）
    if flags["TW"]:
        service_times, time_windows = _tw_fields_full(data["node_coords"])
        data["service_times"] = service_times
        data["time_windows"]  = time_windows

    # O: open 仅语义，不新增字段
    return data

# ----------------------------
# save
# ----------------------------
def _parse_problems_arg(s: str):
    if s.strip().lower() == "all":
        return list_all_canonical()
    return [x.strip() for x in s.split(",") if x.strip()]

def save_all(problems, problem_size, num_instances, out_dir="data", split="test"):
    os.makedirs(out_dir, exist_ok=True)
    for p in problems:
        data = generate_instances(p, problem_size, num_instances)
        out_path = os.path.join(out_dir, f"{p.lower()}{problem_size}_{split}.npz")
        np.savez(out_path, **data)
        print(f"Saved {out_path}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate CVRP/OVRP/VRPB/VRPL/VRPTW and combos (VRPL ~ DCVRP fields)")
    ap.add_argument("--problem_size", type=int, default=100)
    ap.add_argument("--num_instances", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="data1")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--problems", type=str, default="all",
                    help="comma-separated or 'all' "
                         "(e.g., 'CVRP,OVRP,VRPB,VRPL,VRPTW,OVRPB,OVRPL,OVRPTW,VRPBL,OVRPBL,VRPBTW,OVRPBTW,VRPLTW,OVRPLTW,VRPBLTW,OVRPBLTW')")
    args = ap.parse_args()

    problems = _parse_problems_arg(args.problems)
    save_all(problems, args.problem_size, args.num_instances, args.out_dir, args.split)

if __name__ == "__main__":
    main()
