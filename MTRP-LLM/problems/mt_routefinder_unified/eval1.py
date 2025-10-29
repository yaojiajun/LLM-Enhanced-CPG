# -*- coding: utf-8 -*-
"""
Multi-problem Evaluation (RouteFinder / RL4CO) — safe_parse_args version (FAST)

- 每个问题只评测一个数据集 (data/{problem}/test/{size}.npz)
- 不打印日志，仅输出综合平均 cost
- 参数解析使用 safe_parse_args(parser, default_args)
- 加速要点：
  * 模型和 batch 全部放到 CUDA（如果可用）
  * AMP 自动混合精度
  * 在 env.reset 前把 batch.to(device)，让环境也在 GPU 上跑
"""

# --- add project parent to sys.path before any imports ---
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))                           # .../Hercules/problems/mt_routefinder_unified
_PROJECT_PARENT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))     # -> /root/autodl-tmp/yao
if _PROJECT_PARENT not in sys.path:
    sys.path.insert(0, _PROJECT_PARENT)
# ---------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import sys
import shlex
import torch
from typing import Optional, Dict

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE

# ---- speed tricks ----
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True  # 让 cuDNN 选择最优算法

if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# utils
# ----------------------------
def _get_device(device_str: str) -> torch.device:
    if "cuda" in device_str and torch.cuda.is_available():
        # 允许传 cuda 或 cuda:0
        try:
            return torch.device(device_str)
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")

# ----------------------------
# test function
# ----------------------------
@torch.no_grad()
def test(policy, td, env: MTVRPEnv,
         num_augment: int = 1,
         augment_fn: str = "dihedral8",
         num_starts: Optional[int] = None,
         device: torch.device = torch.device("cpu"),
         use_amp: bool = False) -> Dict[str, torch.Tensor]:

    costs_bks = td.get("costs_bks", None)
    with torch.inference_mode():
        if use_amp and device.type == "cuda":
            amp_ctx = torch.amp.autocast("cuda")
        else:
            from contextlib import nullcontext
            amp_ctx = nullcontext()
        with amp_ctx:
            n_start = env.get_num_starts(td) if num_starts is None else num_starts
            n_start = 10*n_start
            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # 原有：还原形状
            reward = unbatchify(out["reward"], (num_augment, n_start))  # (B, A, S)
            actions = out.get("actions", None)
            if actions is not None:
                actions = unbatchify(actions, (num_augment, n_start))    # (B, A, S, T, ...)

            # 第一步：在 starts 维（S，最后一维）取最大，并同步挑出对应 actions
            if n_start > 1:
                max_start_vals, start_idx = reward.max(dim=-1)           # (B, A), (B, A)
                out["max_reward"] = max_start_vals
                if actions is not None:
                    # 在 dim=2（S）上按 start_idx 选动作
                    idx_s = start_idx.unsqueeze(-1)
                    for _ in range(actions.dim() - 3):
                        idx_s = idx_s.unsqueeze(-1)
                    best_over_starts = torch.gather(
                        actions, dim=2,
                        index=idx_s.expand(*actions.shape[:2], 1, *actions.shape[3:])
                    ).squeeze(2)  # (B, A, T, ...)
                else:
                    best_over_starts = None
            else:
                out["max_reward"] = reward                               # (B, A)
                best_over_starts = actions.squeeze(2) if actions is not None else None  # (B, A, T, ...)

            # 第二步：在 augment 维（A，dim=1）取最大，并同步挑出对应 actions
            if num_augment > 1:
                max_aug_reward, aug_idx = out["max_reward"].max(dim=1)   # (B), (B)
                out["max_aug_reward"] = max_aug_reward
                if best_over_starts is not None:
                    idx_a = aug_idx.unsqueeze(-1)
                    for _ in range(best_over_starts.dim() - 2):
                        idx_a = idx_a.unsqueeze(-1)
                    out["best_actions"] = torch.gather(
                        best_over_starts, dim=1,
                        index=idx_a.expand(best_over_starts.shape[0], 1, *best_over_starts.shape[2:])
                    ).squeeze(1)  # (B, T, ...)
            else:
                out["max_aug_reward"] = out["max_reward"]
                if best_over_starts is not None:
                    out["best_actions"] = best_over_starts.squeeze(1)     # (B, T, ...)

            # 原有 gap 计算保持不变
            if costs_bks is not None:
                costs = -out["max_aug_reward"]
                bks = torch.abs(costs_bks)
                out["gap_to_bks"] = 100.0 * (costs - bks) / bks
            else:
                out["gap_to_bks"] = torch.full_like(out["max_aug_reward"], 69420.0)
    return out


def _auto_dataset(problem: str, size: int) -> Optional[str]:
    # path = f"data_train/{problem}/test/{size}_30.npz"
    # path = f"data_real/{problem}/test/{size}.npz"
    # path = f"data_real_first/{problem}/test/{size}_1.npz"
    # path = f"set_data/vrptw/R103.npz"
    path = f"CVRP-LIB_data/cvrp/X-n101-k25.npz"
    return path if os.path.isfile(path) else None

# ----------------------------
# safe_parse_args
# ----------------------------
_KNOWN_FLAGS = {"--checkpoint","--problems","--size","--batch_size","--device"}
_PREFIX_BLACKLIST = ("-f","--ip","--stdin","--control","--hb","--Session","--Kernel")

def _sanitize_argv(argv: list[str]) -> list[str]:
    clean: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if any(tok.startswith(p) for p in _PREFIX_BLACKLIST):
            i += 1
            continue
        if tok in _KNOWN_FLAGS:
            clean.append(tok)
            if tok in {"--checkpoint","--problems","--size","--batch_size","--device"}:
                if i+1 < len(argv):
                    clean.append(argv[i+1]); i+=2; continue
        elif tok.startswith("--") and (tok.split("=")[0] in _KNOWN_FLAGS):
            clean.append(tok)
        i += 1
    return clean

def safe_parse_args(parser: argparse.ArgumentParser, default_args: list[str]):
    env_line = os.environ.get("EVAL_ARGS", "").strip()
    if env_line:
        argv = shlex.split(env_line)
        return parser.parse_args(argv)
    raw = sys.argv[1:]
    argv = _sanitize_argv(raw)
    if not argv:
        return parser.parse_args(default_args)
    return parser.parse_args(argv)

# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/100/rf-pomo.ckpt")
    # 注意：argparse 的 type=str + 默认值为 list 时可能踩坑；这里保持你的原逻辑
    parser.add_argument("--problems", type=str, default=['CVRP','VRPTW','OVRP','VRPB','VRPL','OVRPB','OVRPL','OVRPTW','VRPBL','OVRPBL','VRPBTW','OVRPBTW','VRPLTW','OVRPLTW','VRPBLTW','OVRPBLTW'])
    #['CVRP','VRPTW','OVRP','VRPB','VRPL','OVRPB','OVRPL','OVRPTW','VRPBL','OVRPBL','VRPBTW','OVRPBTW','VRPLTW','OVRPLTW','VRPBLTW','OVRPBLTW']
    parser.add_argument("--size", type=int, default=100) #"cvrp", "vrptw", "ovrp", "vrpb", "vrpl"
    parser.add_argument("--batch_size", type=int, default=1000)  # 原来 1000；GPU 上建议先用 256（按显存再调）
    parser.add_argument("--device", type=str, default="cuda:0")

    # 若没有任何参数，就用 parser 的默认值
    opts = safe_parse_args(parser, [])

    # 解析 problems
    problems_arg = opts.problems
    if isinstance(problems_arg, str):
        problems = [p.strip().lower() for p in problems_arg.split(",")]
    else:
        problems = [p.strip().lower() for p in problems_arg]

    device = _get_device(opts.device)
    use_amp = (device.type == "cuda")

    # 选择模型类型
    ckpt = os.path.basename(opts.checkpoint).lower()
    if "mvmoe" in ckpt:
        BaseLitModule = MVMoE
    elif "mtpomo" in ckpt:
        BaseLitModule = MTPOMO
    elif "moe" in ckpt:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    # 把模型直接加载到目标 device，并移到 eval 模式
    # 替换为 ↓
    model = BaseLitModule.load_from_checkpoint(opts.checkpoint, map_location="cpu", strict=False)
    model.eval()  # 先在 CPU 上构建
    model.to(device)  # 再整体迁移到目标 device
    policy = model.policy.to(device).eval()

    # 环境本身会跟随传入 TensorDict 的 device 工作；因此只要 batch 在 GPU，环境也在 GPU 上算
    env = MTVRPEnv()

    overall_sum, overall_cnt = 0.0, 0
    for problem in problems:
        dataset = _auto_dataset(problem, opts.size)
        if dataset is None:
            continue

        td_test = env.load_data(dataset)
        # 如果你的 get_dataloader 支持 num_workers / pin_memory，可以这样开（否则按原有默认）：
        try:
            dataloader = get_dataloader(td_test, batch_size=opts.batch_size, num_workers=2, pin_memory=True)
        except TypeError:
            dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        res = []
        for batch in dataloader:
            # 关键：先把 batch 搬到 device，再 reset，这样环境内部张量也在同一设备上
            batch = batch.to(device, non_blocking=True)
            td_batch = env.reset(batch)
            o = test(policy, td_batch, env, device=device, use_amp=use_amp)
            res.append(o)

        if not res:
            continue

        out = {"max_aug_reward": torch.cat([o["max_aug_reward"] for o in res], dim=0)}
        cost = -out["max_aug_reward"].mean().item()
        # 直接输出最佳行为（无任何判断/异常处理）
        # best_actions_all = torch.cat([o["best_actions"] for o in res], dim=0)
        # print("best_actions[0]:", best_actions_all[0].tolist())
        print(problem)
        print(cost)
        overall_sum += cost
        overall_cnt += 1

    if overall_cnt > 0:
        print(overall_sum / overall_cnt)  # 只输出综合平均 cost

if __name__ == "__main__":
    # 如果确实在 CPU 上跑，别强行限制到 1 线程；在 GPU 上跑这条也没必要
    # 这里我们不再 set_num_threads(1)
    main()
