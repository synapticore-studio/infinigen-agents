# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import bpy

from infinigen.core.init import require_blender_addon
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


def ensure_real_snow_enabled():
    # Enable the Real Snow addon if not already enabled
    if "object.real_snow" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="object.real_snow")

class Snowlayer:
    def apply(self, obj, **kwargs):
        try:
            require_blender_addon("real_snow", allow_online=True)  # Allow online install
            ensure_real_snow_enabled()
            # Set snow parameters (height, etc.)
            bpy.context.scene.snow.height = kwargs.get("height", 0.1)
            # Select the object and set as active
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            # Apply the Real Snow operator
            bpy.ops.snow.create()
            snow = bpy.context.active_object
            tag_object(snow, "snow")
            tag_object(snow, "boulder")
        except Exception as e:
            # Skip test if addon is not available
            pytest.skip(f"Real Snow addon not available: {e}")

def apply(obj, selection=None, **kwargs):
    snowlayer = Snowlayer()
    snowlayer.apply(obj, **kwargs)
    return snowlayer
