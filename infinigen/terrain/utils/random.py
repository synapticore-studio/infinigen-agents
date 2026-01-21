# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import POINTER, c_float, c_int32, c_size_t

import bpy
import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.core.util.random import random_int, random_int_large, random_nat
from .ctype_util import ASFLOAT, load_cdll


def chance(x):
    return np.random.uniform() < x


def perlin_noise(positions, device, freq, octaves, seed):
    dll = load_cdll(f"terrain/lib/{device}/utils/FastNoiseLite.so")
    func = dll.perlin_call
    func.argtypes = [
        c_size_t,
        POINTER(c_float),
        POINTER(c_float),
        c_int32,
        c_int32,
        c_float,
    ]
    func.restype = None
    values = np.zeros(len(positions), dtype=np.float32)
    func(
        len(positions),
        ASFLOAT(AC(positions.astype(np.float32))),
        ASFLOAT(values),
        seed,
        octaves,
        freq,
    )
    del dll
    return values


def drive_param(parameter, scale=1, offset=0, index=None, name="default_value"):
    """Create driver with modern Blender 4.5.3+ features"""
    try:
        driver = parameter.driver_add(name)
        if index is not None:
            driver = driver[index]
        driver.driver.expression = f"frame*{scale}+{offset}"

        # Modern view layer update
        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Warning: Could not create driver: {e}")
