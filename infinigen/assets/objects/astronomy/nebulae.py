# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Nebulae and Gas Clouds for Infinigen

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
class EmissionNebulaFactory(AssetFactory):
    """Generate emission nebulae (glowing gas clouds)"""

    def __init__(self, factory_seed, nebula_size=50.0, density=0.5, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.nebula_size = nebula_size
        self.density = density

    def create_emission_nebula_geometry(self, nw: NodeWrangler):
        """Create emission nebula geometry with glowing effects"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create main nebula structure
        nebula_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.1, 0.3),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.7, 0.9),
            },
        )

        # Create additional turbulence
        turbulence = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.2, 0.6),
                "Detail": uniform(6, 10),
                "Dimension": uniform(0.1, 0.4),
            },
        )

        # Create wisps and tendrils
        wisps = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 1.5),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Combine all noise patterns
        nebula_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nebula_noise.outputs["Fac"],
                1: turbulence,
            },
            attrs={"operation": "MULTIPLY"},
        )

        final_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nebula_structure,
                1: wisps.outputs["Distance"],
            },
            attrs={"operation": "ADD"},
        )

        # Create density field
        density_field = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: final_structure,
                1: nw.new_value(self.density, "density_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create emission nebula asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.nebula_size, subdivisions=2)
        nebula = bpy.context.active_object
        nebula.name = f"EmissionNebula_{self.factory_seed}"

        # Add geometry nodes for nebula
        surface.add_geomod(nebula, self.create_emission_nebula_geometry, apply=True)

        return nebula


@gin.configurable
class ReflectionNebulaFactory(AssetFactory):
    """Generate reflection nebulae (dust clouds reflecting starlight)"""

    def __init__(self, factory_seed, nebula_size=30.0, dust_density=0.3, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.nebula_size = nebula_size
        self.dust_density = dust_density

    def create_reflection_nebula_geometry(self, nw: NodeWrangler):
        """Create reflection nebula geometry with dust particles"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create dust cloud structure
        dust_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.2, 0.8),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.5, 0.8),
            },
        )

        # Create dust clumps
        dust_clumps = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(1.0, 3.0),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Combine dust patterns
        dust_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: dust_noise.outputs["Fac"],
                1: dust_clumps.outputs["Distance"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create density field
        density_field = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: dust_structure,
                1: nw.new_value(self.dust_density, "dust_density_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create reflection nebula asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.nebula_size, subdivisions=2)
        nebula = bpy.context.active_object
        nebula.name = f"ReflectionNebula_{self.factory_seed}"

        # Add geometry nodes for nebula
        surface.add_geomod(nebula, self.create_reflection_nebula_geometry, apply=True)

        return nebula


@gin.configurable
class DarkNebulaFactory(AssetFactory):
    """Generate dark nebulae (dust clouds blocking light)"""

    def __init__(self, factory_seed, nebula_size=40.0, opacity=0.8, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.nebula_size = nebula_size
        self.opacity = opacity

    def create_dark_nebula_geometry(self, nw: NodeWrangler):
        """Create dark nebula geometry with dense dust"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create dense dust structure
        dust_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.3, 1.0),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.6, 0.9),
            },
        )

        # Create dust filaments
        filaments = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 1.5),
                "Detail": uniform(6, 10),
                "Dimension": uniform(0.1, 0.3),
            },
        )

        # Combine dust patterns
        dust_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: dust_noise.outputs["Fac"],
                1: filaments,
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create opacity field
        opacity_field = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: dust_structure,
                1: nw.new_value(self.opacity, "opacity_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create dark nebula asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.nebula_size, subdivisions=2)
        nebula = bpy.context.active_object
        nebula.name = f"DarkNebula_{self.factory_seed}"

        # Add geometry nodes for nebula
        surface.add_geomod(nebula, self.create_dark_nebula_geometry, apply=True)

        return nebula


@gin.configurable
class PlanetaryNebulaFactory(AssetFactory):
    """Generate planetary nebulae (expanding gas shells)"""

    def __init__(
        self, factory_seed, nebula_size=20.0, shell_thickness=0.3, coarse=False
    ):
        super().__init__(factory_seed, coarse=coarse)
        self.nebula_size = nebula_size
        self.shell_thickness = shell_thickness

    def create_planetary_nebula_geometry(self, nw: NodeWrangler):
        """Create planetary nebula geometry with expanding shell"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create shell structure
        shell_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 2.0),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.4, 0.7),
            },
        )

        # Create expansion patterns
        expansion = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(1.0, 3.0),
                "Detail": uniform(4, 8),
                "Dimension": uniform(0.1, 0.4),
            },
        )

        # Combine shell patterns
        shell_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: shell_noise.outputs["Fac"],
                1: expansion,
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create thickness field
        thickness_field = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: shell_structure,
                1: nw.new_value(self.shell_thickness, "thickness_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create planetary nebula asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.nebula_size, subdivisions=2)
        nebula = bpy.context.active_object
        nebula.name = f"PlanetaryNebula_{self.factory_seed}"

        # Add geometry nodes for nebula
        surface.add_geomod(nebula, self.create_planetary_nebula_geometry, apply=True)

        return nebula


@gin.configurable
class SupernovaRemnantFactory(AssetFactory):
    """Generate supernova remnants (expanding shock waves)"""

    def __init__(
        self, factory_seed, remnant_size=100.0, shock_strength=0.7, coarse=False
    ):
        super().__init__(factory_seed, coarse=coarse)
        self.remnant_size = remnant_size
        self.shock_strength = shock_strength

    def create_supernova_remnant_geometry(self, nw: NodeWrangler):
        """Create supernova remnant geometry with shock waves"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create shock wave structure
        shock_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.1, 0.5),
                "Detail": uniform(10, 15),
                "Roughness": uniform(0.8, 1.0),
            },
        )

        # Create expansion rings
        rings = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.2, 0.8),
                "Detail": uniform(6, 10),
                "Dimension": uniform(0.1, 0.3),
            },
        )

        # Create filamentary structure
        filaments = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 2.0),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Combine all patterns
        shock_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: shock_noise.outputs["Fac"],
                1: rings,
            },
            attrs={"operation": "MULTIPLY"},
        )

        final_structure = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: shock_structure,
                1: filaments.outputs["Distance"],
            },
            attrs={"operation": "ADD"},
        )

        # Create shock strength field
        shock_field = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: final_structure,
                1: nw.new_value(self.shock_strength, "shock_strength_factor"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create supernova remnant asset"""
        bpy.ops.mesh.primitive_ico_sphere_add(radius=self.remnant_size, subdivisions=2)
        remnant = bpy.context.active_object
        remnant.name = f"SupernovaRemnant_{self.factory_seed}"

        # Add geometry nodes for remnant
        surface.add_geomod(remnant, self.create_supernova_remnant_geometry, apply=True)

        return remnant
