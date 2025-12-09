# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Satellite SimObject for Infinigen

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
    "nodegroup_satellite_body", singleton=False, type="GeometryNodeTree"
)
def nodegroup_satellite_body(nw: NodeWrangler, **kwargs):
    """Create the main body of the satellite"""

    # Main body (cubic)
    main_body = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": (1.0, 1.0, 1.0),
        },
    )

    # Communication dish
    dish = nw.new_node(
        Nodes.MeshCylinder,
        input_kwargs={
            "Vertices": 16,
            "Radius": 0.8,
            "Depth": 0.1,
        },
    )

    # Position dish on top
    positioned_dish = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": dish,
            "Offset": (0, 0, 0.6),
        },
    )

    # Antennas
    antenna = nw.new_node(
        Nodes.MeshCylinder,
        input_kwargs={
            "Vertices": 8,
            "Radius": 0.02,
            "Depth": 1.0,
        },
    )

    # Position antennas
    antenna_positions = [(0.5, 0, 0.5), (-0.5, 0, 0.5), (0, 0.5, 0.5), (0, -0.5, 0.5)]

    antenna_geometries = []
    for pos in antenna_positions:
        positioned_antenna = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": antenna,
                "Offset": pos,
            },
        )
        antenna_geometries.append(positioned_antenna)

    all_antennas = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": antenna_geometries}
    )

    # Join all parts
    final_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [main_body, positioned_dish, all_antennas]},
    )

    return final_geometry


@node_utils.to_nodegroup(
    "nodegroup_satellite_solar_panels", singleton=False, type="GeometryNodeTree"
)
def nodegroup_satellite_solar_panels(nw: NodeWrangler, **kwargs):
    """Create solar panels for the satellite"""

    # Solar panel
    solar_panel = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": (0.05, 2.0, 1.0),
        },
    )

    # Create two panels (left and right)
    left_panel = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": solar_panel,
            "Offset": (-0.6, 0, 0),
        },
    )

    right_panel = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": solar_panel,
            "Offset": (0.6, 0, 0),
        },
    )

    # Join panels
    both_panels = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [left_panel, right_panel]}
    )

    return both_panels


@gin.configurable
class SatelliteFactory(AssetFactory):
    """Factory for satellite sim objects"""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.sim_blueprint = blueprint_path_completion("satellite.json")

    def create_asset(self, **kwargs):
        """Create satellite asset"""
        # Create main body (simplified)
        body_obj = butil.spawn_cube(size=1)
        body_obj.name = "satellite_body"

        # Create solar panels (simplified)
        panels_obj = butil.spawn_cube(size=1)
        panels_obj.name = "satellite_panels"
        panels_obj.parent = body_obj
        panels_obj.location = (2, 0, 0)

        # Apply materials (simplified)
        body_material = bpy.data.materials.new(name="SatelliteBody")
        body_material.use_nodes = True
        body_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.7,
            0.7,
            0.8,
            1.0,
        )  # Light gray

        panels_material = bpy.data.materials.new(name="SatellitePanels")
        panels_material.use_nodes = True
        panels_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.0,
            0.0,
            0.2,
            1.0,
        )  # Dark blue

        body_obj.data.materials.append(body_material)
        panels_obj.data.materials.append(panels_material)

        return body_obj

    def spawn_asset(self, i=0, **kwargs):
        """Spawn asset for simulation"""
        return self.create_asset(**kwargs)
        return self.create_asset(**kwargs)
