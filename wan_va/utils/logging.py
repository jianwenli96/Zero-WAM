# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

logger = logging.getLogger()


def add_file_logger(save_root, rank):
    """Keep each distributed worker's logs in its own file."""
    log_dir = Path(save_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = 'train.log' if rank == 0 else f'train_rank_{rank}.log'
    handler = logging.FileHandler(log_dir / filename, encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s - rank {rank} - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    return handler


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
