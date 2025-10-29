# -*- coding: utf-8 -*-
"""
Single-problem evaluator aligned with the multi-problem evaluator conventions
- Uses the same arg names / file layout: args.problems (list), args.test_datasets (list)
- Uses the same pretrained weight path: pretrained/multi.best
"""
import os
import sys
import argparse
import time

from args import add_common_args, add_common_training_args  # 若下游会读取训练开关，保留注入
from learning.data_iterators import DataIterator
from learning.tester import Tester
from utils.exp import setup_experiment, setup_test_environment

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def ensure_file(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} 不存在: {path}")

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_common_training_args(parser)  # 与多问题脚本保持一致，可由命令行覆盖
    args, _ = parser.parse_known_args()

    # ===== 与多问题评测器保持一致的配置 =====
    problem_name  = 'cvrptw'                      # 小写
    dataset_path  = 'data/cvrptw100_test.npz'     # 与多问题脚本中一致
    args.problems = [problem_name]                # 列表
    args.test_datasets = [dataset_path]           # 列表
    args.pretrained_model = 'pretrained/multi.best'  # 统一权重

    # 基础实验目录等初始化（与多问题脚本一致）
    setup_experiment(args)

    # 路径健壮性校验
    ensure_file(dataset_path, f"{problem_name} 测试数据")
    ensure_file(args.pretrained_model, "预训练模型")

    # 构建模型/环境
    net = setup_test_environment(args)

    # 准备数据 & Tester
    data_iterator = DataIterator(args, ddp=False)
    if not hasattr(data_iterator, "test_datasets"):
        raise AttributeError("DataIterator 缺少属性 test_datasets（请核对类实现/字段名）")
    tester = Tester(args, net, data_iterator.test_datasets)

    # 加载权重
    if not hasattr(tester, "load_model"):
        raise AttributeError("Tester 缺少方法 load_model（请核对类实现）")
    tester.load_model(args.pretrained_model)

    # 评测
    start_time = time.time()
    avg_obj_dict = tester.test()   # 与多问题评测器相同的返回结构
    elapsed = time.time() - start_time

    # 统一打印方式（与多问题脚本一致）
    key = problem_name.lower()
    val = None
    if isinstance(avg_obj_dict, dict) and key in avg_obj_dict and 'objectives' in avg_obj_dict[key]:
        val = avg_obj_dict[key]['objectives']
    print(f"[*] {problem_name.upper()} objective: {val}")
    print("inference time", elapsed)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        raise
