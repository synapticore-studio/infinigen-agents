# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import os

import gin
import psutil

from infinigen.core.util.logging import Timer as oTimer


def report_memory():
    """Report memory usage with modern error handling"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Warning: Could not report memory usage: {e}")


@gin.configurable("TerrainTimer")
class Timer(oTimer):
    def __init__(self, desc, verbose):
        super().__init__(desc)
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        if self.verbose:
            super().__exit__(exc_type, exc_val, traceback)
            report_memory()
