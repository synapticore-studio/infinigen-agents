# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Space Station SimObject for Infinigen

import functools

import bpy
import gin
import numpy as np

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import metal, plastic
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.paths import blueprint_path_completion
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "nodegroup_space_station_core", singleton=False, type="GeometryNodeTree"
)
def nodegroup_space_station_core(nw: NodeWrangler, **kwargs):
    """Create the central core of the space station"""

    # Main cylindrical core
    core_cylinder = nw.new_node(
        Nodes.MeshCylinder,
        input_kwargs={
            "Vertices": 32,
            "Radius": 2.0,
            "Depth": 8.0,
        },
    )

    # Add docking ports
    docking_port = nw.new_node(
        Nodes.MeshCylinder,
        input_kwargs={
            "Vertices": 16,
            "Radius": 0.5,
            "Depth": 1.0,
        },
    )

    # Position docking ports around the core
    docking_positions = [(2.5, 0, 0), (-2.5, 0, 0), (0, 2.5, 0), (0, -2.5, 0)]

    docking_geometries = []
    for pos in docking_positions:
        positioned_docking = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": docking_port,
                "Offset": pos,
            },
        )
        docking_geometries.append(positioned_docking)

    # Join all geometries
    all_docking = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": docking_geometries}
    )

    final_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [core_cylinder, all_docking]}
    )

    return final_geometry


@node_utils.to_nodegroup(
    "nodegroup_space_station_solar_panels", singleton=False, type="GeometryNodeTree"
)
def nodegroup_space_station_solar_panels(nw: NodeWrangler, **kwargs):
    """Create solar panels for the space station"""

    # Solar panel geometry
    solar_panel = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": (0.1, 4.0, 2.0),
        },
    )

    # Create multiple panels
    panel_positions = [
        (3.0, 0, 0),
        (-3.0, 0, 0),
        (0, 3.0, 0),
        (0, -3.0, 0),
        (2.1, 2.1, 0),
        (-2.1, 2.1, 0),
        (2.1, -2.1, 0),
        (-2.1, -2.1, 0),
    ]

    panel_geometries = []
    for pos in panel_positions:
        positioned_panel = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": solar_panel,
                "Offset": pos,
            },
        )
        panel_geometries.append(positioned_panel)

    # Join all panels
    all_panels = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": panel_geometries}
    )

    return all_panels


@gin.configurable
class SpaceStationFactory(AssetFactory):
    """Factory for space station sim objects"""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.sim_blueprint = blueprint_path_completion("space_station.json")

    def create_asset(self, **kwargs):
        """Create space station asset"""
        # Create main core (simplified)
        core_obj = butil.spawn_cube(size=2)
        core_obj.name = "space_station_core"

        # Create solar panels (simplified)
        panels_obj = butil.spawn_cube(size=1)
        panels_obj.name = "space_station_panels"
        panels_obj.parent = core_obj
        panels_obj.location = (3, 0, 0)

        # Apply materials (simplified)
        core_material = bpy.data.materials.new(name="SpaceStationCore")
        core_material.use_nodes = True
        core_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.8,
            0.8,
            0.9,
            1.0,
        )  # Silver

        panels_material = bpy.data.materials.new(name="SpaceStationPanels")
        panels_material.use_nodes = True
        panels_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.1,
            0.1,
            0.1,
            1.0,
        )  # Dark

        core_obj.data.materials.append(core_material)
        panels_obj.data.materials.append(panels_material)

        return core_obj

    def spawn_asset(self, i=0, **kwargs):
        """Spawn asset for simulation"""
        return self.create_asset(**kwargs)
        return self.create_asset(**kwargs)
