# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import gin
import numpy as np

from infinigen.core.util.organization import Materials, Transparency
from infinigen.terrain.elements.core import Element


@gin.configurable
class Ground(Element):
    """Ground element for terrain generation"""
    
    def __init__(self, device, caves=None, scale=1.0, height=0.0, with_sand_dunes=False):
        super().__init__(
            lib_name="ground",
            material=Materials.Ground,
            transparency=Transparency.Opaque
        )
        self.device = device
        self.caves = caves
        self.scale = scale
        self.height = height
        self.with_sand_dunes = with_sand_dunes
        
    def __call__(self, *args, **kwargs):
        """Call the ground element"""
        return super().__call__(*args, **kwargs)
