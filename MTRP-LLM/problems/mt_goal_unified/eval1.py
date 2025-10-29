# -*- coding: utf-8 -*-
"""
Multi-problem evaluator for GOAL-CO
- Loops over a list of problem types (e.g., CVRP, VRPTW, OVRP, VRPB, VRPL)
- For each problem, sets dataset + env, runs Tester, and collects results
"""
import os
import argparse
import time
from args import add_common_args, add_common_training_args  # 如果没用到训练参数，这行可保留无影响
from learning.data_iterators import DataIterator
from learning.tester import Tester
from utils.exp import setup_experiment, setup_test_environment

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_one_problem(args, problem_name: str, dataset_path: str):
    """Run evaluation for a single problem, return (avg_obj, raw_dict)."""
    # 1) 逐问题地设置 args
    args.problems = [problem_name.lower()]         # e.g., 'cvrp', 'vrptw', ...
    args.test_datasets = [dataset_path]            # e.g., 'data/cvrp100_test.npz'
    # 你也可以在这里针对不同问题设置额外的参数（如 tw/d/open 标志），
    # 但通常数据文件里已经编码了这些约束特征。

    # 2) 环境与网络（注意：每个问题都可以复用同一套预训练模型）
    net = setup_test_environment(args)

    # 3) 数据迭代器 & Tester
    data_iterator = DataIterator(args, ddp=False)
    tester = Tester(args, net, data_iterator.test_datasets)

    # 4) 加载多任务预训练模型（或按需切换权重）
    tester.load_model(args.pretrained_model)

    # 5) 执行评测
    start = time.time()
    avg_obj = tester.test()     # -> dict: {problem_name: {'objectives': tensor/float, ...}, ...}
    elapsed = time.time() - start

    return avg_obj, elapsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # add_common_training_args(parser)  # 如果不需要，注释掉也行
    args, _ = parser.parse_known_args()

    # ======== 配置区（按需修改）========
    # 评测的问题类型（名称用你们工程里 Tester/环境识别的写法；通常小写）
    problem_names = [ 'cvrptw', 'ocvrp', 'bcvrp', 'dcvrp']

    # 每个问题对应的数据文件路径（示例：与 cvrp 相同的 npz 命名规范）
    # 如果你的数据文件名不同，请在这里改对。
    dataset_map = {
        # 'cvrp':  'data/cvrp100_test.npz',
        'cvrptw': 'data/cvrptw100_test.npz',
        'ocvrp':  'data/ocvrp100_test.npz',
        'bcvrp':  'data/bcvrp100_test.npz',
        'dcvrp':  'data/dcvrp100_test.npz',
    }

    # 统一使用的多任务权重
    args.pretrained_model = "pretrained/multi.best"

    # 其它全局评测参数（按需从命令行覆盖）
    # 例如 batch_size、pomo_size、device、knns、beam_size 等都可以在 args 里设置
    # 这里不强行改写，保留命令行/默认值

    # ======== 实际评测 ========
    setup_experiment(args)

    all_vals = {}
    overall_sum = 0.0
    overall_cnt = 0
    total_time = 0.0

    for name in problem_names:
        if name.lower() not in dataset_map:
            print(f"[!] No dataset configured for problem '{name}', skip.")
            continue

        dataset_path = dataset_map[name.lower()]
        if not os.path.exists(dataset_path):
            print(f"[!] Dataset not found: {dataset_path}, skip '{name}'.")
            continue

        print(f"\n=== Evaluating {name.upper()} on {dataset_path} ===")
        avg_obj_dict, t = run_one_problem(args, name, dataset_path)
        total_time += t

        # Tester.test() 的返回结构通常是：{ 'cvrp': {'objectives': value, ...}, ... }
        # 为了稳妥，统一按小写键名取值
        key = name.lower()
        if key in avg_obj_dict and 'objectives' in avg_obj_dict[key]:
            val = avg_obj_dict[key]['objectives']
        else:
            # 兜底：如果 Tester 返回形式不同，这里打印原始字典帮助定位
            print(f"[?] Unexpected result structure for {name}: {avg_obj_dict}")
            val = None

        print(f"[*] {name.upper()} objective: {val}")
        all_vals[name.upper()] = val

        # 聚合（仅对可转 float 的值做整体平均）
        try:
            overall_sum += float(val)
            overall_cnt += 1
        except Exception:
            pass

    print("\n=== Summary ===")
    for k, v in all_vals.items():
        print(f"{k}: {v}")
    if overall_cnt > 0:
        overall_avg = overall_sum / overall_cnt
        print(f"[*] Overall Average over {overall_cnt} problems: {overall_avg}")
    else:
        print("[*] Overall Average: N/A (no numeric results)")

