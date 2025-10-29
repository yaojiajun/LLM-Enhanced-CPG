"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from learning import decoding_fn
from utils.data_manipulation import prepare_batch
from utils.misc import get_opt_gap, EpochMetrics
DEBUG_NUM_BATCHES = 1


def validate_model(net, datasets, epoch_done, logger, debug, beam_size=100, knns=-1, what="val"):
    metrics_per_problem = dict()
    net.eval()
    with torch.no_grad():
        for problem_name, dataloader in datasets.items():
            current_dataset_epoch_metrics = EpochMetrics()
            for batch_num, batch in enumerate(dataloader):
                metrics = get_minibatch_val_test_metrics(net, beam_size, knns, batch, problem_name)
                current_dataset_epoch_metrics.update(metrics)
                if batch_num == DEBUG_NUM_BATCHES and debug:
                    break
            # 取均值
            means = current_dataset_epoch_metrics.get_means()
            dataset_res = {f'{what}_{k}_{problem_name}': v for k, v in means.items()}

            # 存 opt_gap 和目标值（比如 obj 或 cost）
            metrics_per_problem[problem_name] = {
                "opt_gap": means.get("opt_gap", None),
                "objectives": means.get("objectives", means.get("cost", None))
            }
            if logger is not None:
                logger.record(dataset_res, epoch_done)
            # else:
            #     print(" ".join([" {}: {:.5f}".format(k, v) for k, v in dataset_res.items()]))

    return metrics_per_problem


def get_minibatch_val_test_metrics(net, beam_size, knns, data, problem_name):
    # autoregressive decoding
    decoding_metrics = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problem_data = prepare_batch(data, problem_name, device, sample=False)

    net.eval()
    with torch.no_grad():
        gt_values = problem_data[-1]
        predicted_values, _ = decoding_fn[problem_name](problem_name, problem_data, net, beam_size, knns)

    opt_gap = get_opt_gap(predicted_values, gt_values, problem_name)
    decoding_metrics.update({"opt_gap": opt_gap})
    decoding_metrics.update({"objectives": predicted_values.float().mean().item()})

    return {**decoding_metrics}
