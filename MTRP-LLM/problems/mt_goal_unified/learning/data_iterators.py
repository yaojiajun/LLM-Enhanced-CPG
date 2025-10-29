"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

from argparse import Namespace
from learning.tsp.dataset import load_dataset as load_tsp_dataset
from learning.pctsp.dataset import load_dataset as load_pctsp_dataset
from learning.cvrp.dataset import load_dataset as load_cvrp_dataset
from learning.cvrptw.dataset import load_dataset as load_cvrptw_dataset
from learning.op.dataset import load_dataset as load_op_dataset
from learning.kp.dataset import load_dataset as load_kp_dataset
from learning.mvc.dataset import load_dataset as load_mvc_dataset
from learning.mis.dataset import load_dataset as load_mis_dataset
from learning.upms.dataset import load_dataset as load_upms_dataset
from learning.jssp.dataset import load_dataset as load_jssp_dataset
from learning.mclp.dataset import load_dataset as load_mclp_dataset


class DataIterator:

    def __init__(self, args: Namespace, ddp: bool = False):

        loaders = {
            'tsp': load_tsp_dataset,
            'trp': load_tsp_dataset,
            'sop': load_tsp_dataset,
            'pctsp': load_pctsp_dataset,

            # VRP 家族（含 o/b/d/tw 组合）
            'cvrp': load_cvrp_dataset,
            'sdcvrp': load_cvrp_dataset,
            'ocvrp': load_cvrp_dataset,
            'dcvrp': load_cvrp_dataset,
            'bcvrp': load_cvrp_dataset,
            'obcvrp': load_cvrp_dataset,
            'odcvrp': load_cvrp_dataset,
            'bdcvrp': load_cvrp_dataset,
            'obdcvrp': load_cvrp_dataset,

            'cvrptw': load_cvrp_dataset,
            'ocvrptw': load_cvrp_dataset,
            'bcvrptw': load_cvrp_dataset,
            'obcvrptw': load_cvrp_dataset,
            'dcvrptw': load_cvrp_dataset,
            'odcvrptw': load_cvrp_dataset,
            'bdcvrptw': load_cvrp_dataset,
            'obdcvrptw': load_cvrp_dataset,

            'op': load_op_dataset,
            'mclp': load_mclp_dataset,
            'kp': load_kp_dataset,
            'mvc': load_mvc_dataset,
            'mis': load_mis_dataset,
            'upms': load_upms_dataset,
            'jssp': load_jssp_dataset,
            'ossp': load_jssp_dataset,
        }

        self.train_datasets = dict()
        if "train_datasets" in args and args.train_datasets is not None:
            assert len(args.problems) == len(args.train_datasets)
            for problem_no, problem in enumerate(args.problems):
                self.train_datasets[problem] = loaders[problem](args.train_datasets[problem_no],
                                                                args.train_batch_size, args.train_datasets_size,
                                                                True, True, "train", ddp)
        else:
            for problem_no, problem in enumerate(args.problems):
                self.train_datasets[problem] = None

        if "val_datasets" in args and args.val_datasets is not None:
            assert len(args.problems) == len(args.val_datasets)
            self.val_datasets = dict()
            for problem_no, problem in enumerate(args.problems):
                self.val_datasets[problem] = loaders[problem](args.val_datasets[problem_no], args.val_batch_size,
                                                              args.val_datasets_size, False, False, "val", False)
        assert len(args.problems) == len(args.test_datasets)
        self.test_datasets = dict()
        for problem_no, problem in enumerate(args.problems):
            self.test_datasets[problem] = loaders[problem](args.test_datasets[problem_no], args.test_batch_size,
                                                           args.test_datasets_size, False, False, "test", False)
