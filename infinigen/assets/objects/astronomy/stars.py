# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Stars and Nebulae for Infinigen

import bpy
import gin
import numpy as np
from mathutils import Vector
from numpy.random import normal, uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import SurfaceTypes


@gin.configurable
class StarFieldFactory(AssetFactory):
    """Generate star fields and constellations"""

    def __init__(self, factory_seed, star_count=1000, radius=100.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.star_count = star_count
        self.radius = radius

    def create_star_field_geometry(self, nw: NodeWrangler):
        """Create star field with procedural stars"""
        group_input = nw.new_node(Nodes.GroupInput)

        # Create random points for stars using simpler approach
        random_value = nw.new_node(
            Nodes.RandomValue,
            input_kwargs={
                "Min": -self.radius,
                "Max": self.radius,
            },
        )

        # Create individual star geometry
        star_geometry = self.create_star_geometry(nw)

        # For now, just return the base geometry
        # TODO: Implement proper star field distribution
        return group_input.outputs["Geometry"]

    def create_star_geometry(self, nw: NodeWrangler):
        """Create individual star geometry"""
        # For now, just return the base geometry
        # TODO: Implement proper star geometry
        group_input = nw.new_node(Nodes.GroupInput)
        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create star field asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.radius, subdivisions=3)
        star_field = bpy.context.active_object
        star_field.name = f"StarField_{self.factory_seed}"

        # Add geometry nodes for star field
        surface.add_geomod(star_field, self.create_star_field_geometry, apply=True)

        return star_field


@gin.configurable
class NebulaFactory(AssetFactory):
    """Generate nebulae and gas clouds"""

    def __init__(self, factory_seed, nebula_type="emission", radius=50.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.nebula_type = nebula_type
        self.radius = radius

    def create_nebula_geometry(self, nw: NodeWrangler):
        """Create nebula geometry with volume effects"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create nebula noise
        nebula_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.1, 0.5),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.7, 0.9),
            },
        )

        # Create additional turbulence
        turbulence = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.2, 0.8),
                "Detail": uniform(6, 10),
                "Dimension": uniform(0.1, 0.4),
            },
        )

        # Combine for nebula structure
        nebula_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nebula_noise.outputs["Fac"],
                1: turbulence,
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create volume density
        volume_density = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nebula_structure,
                1: nw.new_value(0.5, "density_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create nebula asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.radius, subdivisions=2)
        nebula = bpy.context.active_object
        nebula.name = f"Nebula_{self.factory_seed}"

        # Add geometry nodes for nebula
        surface.add_geomod(nebula, self.create_nebula_geometry, apply=True)

        return nebula


@gin.configurable
class GalaxyFactory(AssetFactory):
    """Generate spiral galaxies"""

    def __init__(self, factory_seed, galaxy_type="spiral", radius=100.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.galaxy_type = galaxy_type
        self.radius = radius

    def create_galaxy_geometry(self, nw: NodeWrangler):
        """Create galaxy geometry with spiral arms"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create spiral arm pattern
        spiral_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.05, 0.2),
                "Detail": uniform(10, 15),
            },
        )

        # Create galactic center
        center_distance = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: position,
                1: nw.new_value(0.0, "center"),
            },
            attrs={"operation": "DISTANCE"},
        )

        # Combine for galaxy structure
        galaxy_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: spiral_noise.outputs["Fac"],
                1: center_distance.outputs["Value"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create galaxy asset"""
        bpy.ops.mesh.primitive_plane_add(size=self.radius * 2, location=(0, 0, 0))
        galaxy = bpy.context.active_object
        galaxy.name = f"Galaxy_{self.factory_seed}"

        # Add geometry nodes for galaxy
        surface.add_geomod(galaxy, self.create_galaxy_geometry, apply=True)

        return galaxy
