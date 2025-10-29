# -*- coding: utf-8 -*-
"""
Multi-problem evaluator for GOAL-CO (FAST)
- 只加载一次模型；复用同一个 Tester / net
- 每个 problem 仅切换 test_datasets
- 最小输出：逐问题 objective + overall 平均
"""
import os
import argparse
import time
import torch

from args import add_common_args, add_common_training_args  # 若没用到也可保留
from learning.data_iterators import DataIterator
from learning.tester import Tester
from utils.exp import setup_experiment, setup_test_environment

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# 速度相关：合理的后端设置
# -----------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True  # 让 cuDNN 选择更快的 kernel

def _to_device_str(dev: str) -> str:
    if "cuda" in dev and torch.cuda.is_available():
        return dev
    return "cpu"

def _set_amp_default(args):
    # 如果 args 没有 use_amp 字段，这里默认为 True（仅在 CUDA 上）
    if not hasattr(args, "use_amp"):
        setattr(args, "use_amp", torch.cuda.is_available())
    else:
        # 如果传了 use_amp，但设备不是 CUDA，就关掉
        if not torch.cuda.is_available():
            args.use_amp = False

def _build_iterator(args, problem_name: str, dataset_path: str):
    # 只改这两项，其余 args 保持不变
    args.problems = [problem_name.lower()]
    args.test_datasets = [dataset_path]
    return DataIterator(args, ddp=False)

def run_eval_over_problems(args, problem_names, dataset_map):
    """加载一次模型，循环评测多个 problem。"""
    device = _to_device_str(getattr(args, "device", "cuda:0"))
    _set_amp_default(args)

    # 1) 实验环境只 setup 一次
    setup_experiment(args)

    # 2) 构建 net / env（不加载权重）
    net = setup_test_environment(args)

    # 3) 构建一个空 tester；紧接着加载一次权重
    #    注意：部分 Tester 的 __init__ 需要 test_datasets；先给个空的占位，再切换
    empty_iterator = _build_iterator(args, problem_names[0], dataset_map[problem_names[0]])
    tester = Tester(args, net, empty_iterator.test_datasets)
    tester.load_model(args.pretrained_model)   # 只调用一次

    # 记录
    results = {}
    overall_sum, overall_cnt, total_time = 0.0, 0, 0.0

    for name in problem_names:
        key = name.lower()
        if key not in dataset_map:
            print(f"[!] No dataset configured for problem '{name}', skip.")
            continue
        dataset_path = dataset_map[key]
        if not os.path.exists(dataset_path):
            print(f"[!] Dataset not found: {dataset_path}, skip '{name}'.")
            continue

        # 4) 切换到该 problem 的测试集
        data_iterator = _build_iterator(args, key, dataset_path)

        # 5) 尝试复用 tester：优先使用 setter；没有就重建一个轻量 tester（不再加载权重）
        if hasattr(tester, "set_test_datasets"):
            tester.set_test_datasets(data_iterator.test_datasets)
        else:
            # 回退方案：新建 Tester，但直接复用已加载好权重的 net
            tester = Tester(args, net, data_iterator.test_datasets)
            # 注意：不要在这里再次调用 tester.load_model(...)

        # 6) 执行评测
        start = time.time()
        avg_obj_dict = tester.test()
        elapsed = time.time() - start
        total_time += elapsed

        # 7) 取数 & 打印
        if key in avg_obj_dict and 'objectives' in avg_obj_dict[key]:
            val = avg_obj_dict[key]['objectives']
        else:
            val = None

        print(f"[*] {name.upper()} objective: {val}")
        results[name.upper()] = val
        try:
            overall_sum += float(val)
            overall_cnt += 1
        except Exception:
            pass

    # 8) 汇总
    if overall_cnt > 0:
        overall_avg = overall_sum / overall_cnt
        print(f"[*] Overall Average over problems:")
        print(overall_avg)
    else:
        print("[*] Overall Average: N/A (no numeric results)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # add_common_training_args(parser)  # 如果不需要训练相关参数，可以注释

    # ======= 可在命令行或这里设置一些常用的默认 =======
    # 统一使用的多任务权重
    default_pretrained = "pretrained/multi.best"
    # 评测的问题列表
    problem_names = ['cvrp', 'cvrptw', 'ocvrp','bcvrp', 'dcvrp', 'obcvrp', 'odcvrp', 'ocvrptw',
        'bdcvrp', 'obdcvrp', 'bcvrptw', 'obcvrptw', 'dcvrptw', 'odcvrptw', 'bdcvrptw', 'obdcvrptw'
    ]

    # problem_names = ['cvrp', 'cvrptw', 'ocvrp', 'bcvrp', 'dcvrp'
    #                                                      'obcvrp', 'odcvrp', 'ocvrptw',
    #                  'bdcvrp', 'obdcvrp', 'bcvrptw', 'obcvrptw', 'dcvrptw', 'odcvrptw', 'bdcvrptw', 'obdcvrptw'
    #                  ]

    # 每个问题对应的数据文件
    # 各问题对应的数据文件（与 export_goal 的命名一致：{name}{size}_{split}.npz）
    # 在 dataset_map 定义的正上方加三行（或直接替换原来的 dataset_map 块）
    DATA_ROOT = os.environ.get("DATA_ROOT", "data_real")  # 可选：数据根目录
    DATA_SIZE = int(os.environ.get("DATA_SIZE", "100"))  # 可选：图规模，默认 50
    DATA_SPLIT = os.environ.get("DATA_SPLIT", "test")  # 可选：split，默认 test

    dataset_map = {
        'cvrp': f'{DATA_ROOT}/cvrp/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'cvrptw': f'{DATA_ROOT}/vrptw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'ocvrp': f'{DATA_ROOT}/ovrp/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'bcvrp': f'{DATA_ROOT}/vrpb/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'dcvrp': f'{DATA_ROOT}/vrpl/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'obcvrp': f'{DATA_ROOT}/ovrpb/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'odcvrp': f'{DATA_ROOT}/ovrpl/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'ocvrptw': f'{DATA_ROOT}/ovrptw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'bdcvrp': f'{DATA_ROOT}/vrpbl/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'obdcvrp': f'{DATA_ROOT}/ovrpbl/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'bcvrptw': f'{DATA_ROOT}/vrpbtw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'obcvrptw': f'{DATA_ROOT}/ovrpbtw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'dcvrptw': f'{DATA_ROOT}/vrpltw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'odcvrptw': f'{DATA_ROOT}/ovrpltw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'bdcvrptw': f'{DATA_ROOT}/vrpbltw/{DATA_SPLIT}/{DATA_SIZE}.npz',
        'obdcvrptw': f'{DATA_ROOT}/ovrpbltw/{DATA_SPLIT}/{DATA_SIZE}.npz',
    }

    args, _ = parser.parse_known_args()
    # 若命令行没指定，则给出更合理的默认
    if not hasattr(args, "pretrained_model") or args.pretrained_model in (None, "", "None"):
        args.pretrained_model = default_pretrained
    if not hasattr(args, "device") or args.device in (None, "", "None"):
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 可选：如果有 batch_size / beam_size / pomo_size 之类的参数，也可以在这里给默认

    run_eval_over_problems(args, problem_names, dataset_map)
