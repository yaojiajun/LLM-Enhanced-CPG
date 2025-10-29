# -*- coding: utf-8 -*-
"""
Unified VRP/VRPTW dataset loader (NO normalization)
- 同时适配：CVRP/VRP 扩展（L/DCVRP、B/backhaul、remaining_distances 等）与 VRPTW（TW/服务时间/出发时间）
- 训练集：启用 collate_func_with_sample（采样 PATH 子问题，生成 *_s 字段）
- 测试/评测：不启用 collate_fn，按全图输出
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from scipy.spatial.distance import pdist, squareform


# ----------------------------
# 小工具
# ----------------------------
def _to_numpy(a):
    if isinstance(a, np.memmap):
        return np.array(a)
    return a

def _as_float32(x):
    return x.astype(np.float32, copy=False)

def _as_int64(x):
    return x.astype(np.int64, copy=False)


# ----------------------------
# 统一 Dataset
# ----------------------------
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):
    """
    同时兼容两类场景：
    - CVRP/VRP(+L/B)：node_coords 或 dist_matrices；node_demands/total_capacities/tour_lens；
      remaining_capacities/via_depots/distance_constraints/remaining_distances(=remaining_distance_constraints)
    - VRPTW：在上面的基础上还可能含有 service_times / time_windows / departure_times
    """
    def __init__(
        self,
        node_coords=None,
        dist_matrices=None,
        node_demands=None,
        total_capacities=None,
        tour_lens=None,
        remaining_capacities=None,
        via_depots=None,
        # L / DCVRP 相关
        distance_constraints=None,
        remaining_distance_constraints=None,
        # TW 相关
        service_times=None,
        time_windows=None,
        departure_times=None,
    ):
        assert (node_coords is not None) ^ (dist_matrices is not None), \
            "Exactly one of (node_coords, dist_matrices) must be provided"

        # 原始数组保存
        self.node_coords = node_coords
        self.dist_matrices = dist_matrices
        self.node_demands = node_demands
        self.total_capacities = total_capacities
        self.tour_lens = tour_lens

        self.remaining_capacities = remaining_capacities
        self.via_depots = via_depots

        self.distance_constraints = distance_constraints
        self.remaining_distance_constraints = remaining_distance_constraints  # aka remaining_distances

        self.service_times = service_times
        self.time_windows = time_windows
        self.departure_times = departure_times

        # 基础长度（以 node_demands 为准）
        assert self.node_demands is not None, "node_demands must be provided"
        self._length = len(self.node_demands)

    def __len__(self):
        return self._length

    def _build_dist(self, item):
        """
        返回 (N,N,2) 的双通道矩阵；必要时从 node_coords 现算。
        不做任何归一化（保持原始尺度）。
        """
        if self.dist_matrices is None:
            # 从坐标现算欧氏距离 → (N,N)
            coords = _to_numpy(self.node_coords[item])
            dmat = squareform(pdist(coords, metric="euclidean"))
        else:
            dmat = _to_numpy(self.dist_matrices[item])
            # 若已是双通道 (N,N,2)，直接返回
            if dmat.ndim == 3 and dmat.shape[-1] == 2:
                return dmat.astype(np.float32, copy=False), 1.0
            # 若是 (N,N)，继续走下面统一成双通道

        # 统一双通道 (N,N,2) = [forward, transpose]
        two_ch = np.stack([dmat, dmat.T], axis=-1).astype(np.float32, copy=False)
        return two_ch, 1.0  # norm_factor 恒为 1.0（未归一化）

    def __getitem__(self, item):
        # 1) 距离矩阵（原始尺度）
        dist_matrices, _ = self._build_dist(item)  # (N,N,2), float32

        # 2) tour_lens（保持原值，不缩放）
        if self.tour_lens is not None:
            tour_lens = np.array(self.tour_lens[item], dtype=np.float32)
        else:
            tour_lens = np.array([], dtype=np.float32)

        # 3) 其余字段（存在就取，不存在就空）
        node_demands = _to_numpy(self.node_demands[item])
        total_capacities = _to_numpy(self.total_capacities[item]) if self.total_capacities is not None else None
        remaining_capacities = _to_numpy(self.remaining_capacities[item]) if self.remaining_capacities is not None else np.array([])
        via_depots = _to_numpy(self.via_depots[item]) if self.via_depots is not None else np.array([])

        # L / DCVRP
        rdc = None
        if self.remaining_distance_constraints is not None:
            rdc_item = _to_numpy(self.remaining_distance_constraints[item])
            if np.isscalar(rdc_item) or isinstance(rdc_item, (np.floating, float, int, np.integer)):
                rdc = np.array([float(rdc_item)], dtype=np.float32)  # 标量安全化
            else:
                rdc = _as_float32(rdc_item)

        dc = None
        if self.distance_constraints is not None:
            dc_item = _to_numpy(self.distance_constraints[item])
            if np.isscalar(dc_item) or isinstance(dc_item, (np.floating, float, int, np.integer)):
                dc = np.array([float(dc_item)], dtype=np.float32)
            else:
                dc = _as_float32(dc_item)

        # TW
        service_times = _to_numpy(self.service_times[item]) if self.service_times is not None else np.array([])
        time_windows = _to_numpy(self.time_windows[item]) if self.time_windows is not None else np.array([])
        departure_times = _to_numpy(self.departure_times[item]) if self.departure_times is not None else np.array([])

        # 4) 打包成张量（保持原始尺度）
        item_dict = DotDict()

        # 距离矩阵
        item_dict.dist_matrices = torch.from_numpy(dist_matrices)            # float32

        # 需求/容量
        item_dict.node_demands = torch.from_numpy(_as_float32(node_demands)) # float32（若你需整型可自行改）
        if total_capacities is not None:
            item_dict.total_capacities = torch.tensor([int(total_capacities)], dtype=torch.int64)
        else:
            item_dict.total_capacities = torch.tensor([], dtype=torch.int64)

        item_dict.remaining_capacities = torch.from_numpy(_as_int64(remaining_capacities)) if remaining_capacities.size else torch.tensor([], dtype=torch.int64)

        # DCVRP 两项
        if dc is not None:
            item_dict.distance_constraints = torch.from_numpy(dc)                            # float32
        if rdc is not None:
            item_dict.remaining_distance_constraints = torch.from_numpy(rdc)                 # float32

        # TW 相关
        if time_windows.size:
            item_dict.time_windows = torch.from_numpy(_as_float32(time_windows))            # (N,2) float32
        if service_times.size:
            item_dict.service_times = torch.from_numpy(_as_float32(service_times))          # (N,)  float32
        if departure_times.size:
            item_dict.departure_times = torch.from_numpy(_as_float32(departure_times))      # (N,)  float32

        # 其它
        item_dict.tour_lens = torch.tensor(tour_lens, dtype=torch.float32)                  # 标量或空
        item_dict.via_depots = torch.from_numpy(_as_int64(via_depots)) if via_depots.size else torch.tensor([], dtype=torch.int64)

        return item_dict


# ----------------------------
# 统一 DataLoader
# ----------------------------
def load_dataset(
    filename,
    batch_size,
    datasets_size=None,
    shuffle=False,
    drop_last=False,
    what="test",
    ddp=False,
):
    data = np.load(filename)

    # 截断长度
    def _sl(key):
        if key in data and datasets_size is not None:
            return data[key][:datasets_size]
        return data[key] if key in data else None

    node_coords = _sl("node_coords")
    dist_matrices = _sl("dist_matrices")

    node_demands = _sl("node_demands")
    total_capacities = _sl("total_capacities")
    tour_lens = _sl("tour_lens")

    remaining_capacities = _sl("remaining_capacities")
    via_depots = _sl("via_depots")

    distance_constraints = _sl("distance_constraints")
    remaining_distance_constraints = _sl("remaining_distances")  # 命名保持兼容你的 CVRP 代码

    service_times = _sl("service_times")
    time_windows = _sl("time_windows")
    departure_times = _sl("departure_times")

    dataset = DataSet(
        node_coords=node_coords,
        dist_matrices=dist_matrices,
        node_demands=node_demands,
        total_capacities=total_capacities,
        tour_lens=tour_lens,
        remaining_capacities=remaining_capacities,
        via_depots=via_depots,
        distance_constraints=distance_constraints,
        remaining_distance_constraints=remaining_distance_constraints,
        service_times=service_times,
        time_windows=time_windows,
        departure_times=departure_times,
    )

    if ddp:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    collate_fn = collate_func_with_sample if what == "train" else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return loader


# ----------------------------
# 统一 collate：采样 PATH 子问题（生成 *_s 字段）
# ----------------------------
def collate_func_with_sample(l_dataset_items):
    """
    采样子问题（PATH）：
    - 对所有条目统一随机 begin_idx
    - 规则：
        * 矩阵类（dist_matrices）：v[begin_idx:, begin_idx:]
        * 一维“仅取当前位置”的（remaining_capacities, departure_times）：v[begin_idx:begin_idx+1]
        * 一维/时间窗等“尾部切片”的（node_demands, via_depots, remaining_distance_constraints, service_times, time_windows）：v[begin_idx:, ...]
        * 标量或长度=1的（total_capacities, distance_constraints, tour_lens）：原样
    最终把采样得到的字段用 {key+'_s': value} 回填，同时保留原字段，便于下游兼容。
    """
    assert len(l_dataset_items) > 0
    # 取 N（节点数）——从 dist_matrices 的第一维
    nb_nodes = l_dataset_items[0].dist_matrices.shape[0]
    # begin in [0, nb_nodes-3], 至少保留 3 个节点
    begin_idx = np.random.randint(0, max(1, nb_nodes - 3))

    l_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            # 标量张量（如 tour_lens 可能是标量张量）直接原样
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                v_s = v
            # 仅“取当前位置”的一维量
            elif k in ("remaining_capacities", "departure_times"):
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] >= begin_idx + 1:
                    v_s = v[begin_idx:begin_idx+1]
                else:
                    v_s = v
            # 矩阵切片（距离矩阵）
            elif k == "dist_matrices":
                # (N,N,2) -> 切前两维
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    v_s = v[begin_idx:, begin_idx:, ...]
                else:
                    v_s = v
            # (N,2) 的时间窗：只切第一维
            elif k == "time_windows":
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    v_s = v[begin_idx:, ...]
                else:
                    v_s = v
            # 尾部切片的一维/二维量
            elif k in ("node_demands", "via_depots", "remaining_distance_constraints", "service_times"):
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    v_s = v[begin_idx:, ...]
                else:
                    v_s = v
            # 标量/长度1的保持不变
            elif k in ("total_capacities", "distance_constraints", "tour_lens"):
                v_s = v
            else:
                # 其它未知字段：尽量按“尾切”处理
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    v_s = v[begin_idx:, ...]
                else:
                    v_s = v

            d_new[k + "_s"] = v_s

        # 合并：保留原字段 + 新字段
        l_new.append({**d, **d_new})

    return default_collate(l_new)
