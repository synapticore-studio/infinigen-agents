# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""
Modern Terrain Rendering Module - Blender 4.5.3+ Features
"""

from .modern_lighting import (
    LightGroups,
    ModernRenderingEngine,
    VirtualShadowMapping,
    setup_modern_terrain_rendering,
)

__all__ = [
    'VirtualShadowMapping',
    'LightGroups',
    'ModernRenderingEngine', 
    'setup_modern_terrain_rendering'
]
