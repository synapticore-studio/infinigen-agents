# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Moons for Infinigen

import math

import bpy
import gin
from mathutils import Vector
from numpy.random import normal, uniform

from infinigen.assets.objects.astronomy.constraints import (
    AstronomicalConstraintSolver,
    OrbitalConstraints,
    RealisticOrbitalMechanics,
)
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import SurfaceTypes


@gin.configurable
class MoonFactory(AssetFactory):
    """Moon factory for planetary satellites"""

    def __init__(
        self,
        factory_seed,
        radius=0.2,
        resolution=32,
        coarse=False,
        parent_planet=None,
        orbit_radius=5.0,
        parent_mass=5.972e24,
        moon_mass=7.342e22,
        use_realistic_orbits=True,
    ):
        super().__init__(factory_seed, coarse=coarse)
        self.radius = radius
        self.resolution = resolution
        self.parent_planet = parent_planet
        self.orbit_radius = orbit_radius
        self.parent_mass = parent_mass
        self.moon_mass = moon_mass
        self.use_realistic_orbits = use_realistic_orbits
        self.planet_type = "moon"

        # Initialize orbital mechanics
        self.orbital_mechanics = RealisticOrbitalMechanics()
        self.constraint_solver = AstronomicalConstraintSolver()

    def create_asset(self, **kwargs):
        """Create moon asset with orbital animation"""
        # Position moon at orbit radius
        initial_position = (self.orbit_radius, 0, 0)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=self.radius, location=initial_position
        )
        moon = bpy.context.active_object
        moon.name = f"Moon_{self.factory_seed}"

        # Add geometry nodes for moon surface
        surface.add_geomod(moon, self.create_moon_geometry, apply=True)

        # Create and apply material
        material = surface.shaderfunc_to_material(self.shader_moon)
        moon.data.materials.append(material)

        # Add orbital animation
        self.add_orbital_animation(moon)

        # Add rotation animation
        self.add_rotation_animation(moon)

        tag_object(moon, SurfaceTypes.AstronomicalMoon)
        return moon

    def create_moon_geometry(self, nw: NodeWrangler):
        """Create moon surface with craters and mountains"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create crater noise
        crater_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(15, 25),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.6, 0.8),
            },
        )

        # Create mountain ranges
        mountain_noise = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(20, 30),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.5, 0.7),
            },
            attrs={"musgrave_dimensions": "3D"},
        )

        # Combine noises for surface displacement
        surface_noise = nw.new_node(
            Nodes.Math,
            input_kwargs={
                "Value": crater_noise.outputs["Fac"],
                "Value_001": mountain_noise.outputs["Fac"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Apply displacement
        displacement = nw.new_node(
            Nodes.Math,
            input_kwargs={
                "Value": surface_noise,
                "Value_001": uniform(0.1, 0.3),
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create vector for offset
        offset_vector = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                "X": displacement,
                "Y": displacement,
                "Z": displacement,
            },
        )

        set_position = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": group_input.outputs["Geometry"],
                "Offset": offset_vector,
            },
        )

        return set_position

    def shader_moon(self, nw: NodeWrangler, random_seed=0):
        """Create shader for moon surface"""
        nw.force_input_consistency()

        with FixedSeed(random_seed):
            position = nw.new_node("ShaderNodeNewGeometry")

            # Create surface noise
            surface_noise = nw.new_node(
                Nodes.NoiseTexture,
                input_kwargs={
                    "Vector": position,
                    "Scale": uniform(10, 15),
                    "Detail": uniform(8, 12),
                },
            )

            # Create crater patterns
            crater_noise = nw.new_node(
                Nodes.VoronoiTexture,
                input_kwargs={
                    "Vector": position,
                    "Scale": uniform(5, 8),
                    "Randomness": uniform(0.7, 0.9),
                },
                attrs={"voronoi_dimensions": "2D"},
            )

            # Create moon colors
            moon_color = nw.new_node(
                Nodes.ColorRamp, input_kwargs={"Fac": surface_noise.outputs["Fac"]}
            )
            moon_color.color_ramp.elements[0].color = (
                uniform(0.4, 0.6),  # Dark gray
                uniform(0.4, 0.6),
                uniform(0.4, 0.6),
                1.0,
            )
            moon_color.color_ramp.elements[1].color = (
                uniform(0.7, 0.9),  # Light gray
                uniform(0.7, 0.9),
                uniform(0.7, 0.9),
                1.0,
            )

            # Add crater effects
            crater_color = nw.new_node(
                Nodes.ColorRamp, input_kwargs={"Fac": crater_noise.outputs["Distance"]}
            )
            crater_color.color_ramp.elements[0].color = (
                0.2,
                0.2,
                0.2,
                1.0,
            )  # Dark craters
            crater_color.color_ramp.elements[1].color = (
                0.8,
                0.8,
                0.8,
                1.0,
            )  # Light areas

            # Mix colors
            final_color = nw.new_node(
                Nodes.MixRGB,
                input_kwargs={
                    "Fac": crater_color.outputs["Color"],
                    "Color1": moon_color.outputs["Color"],
                    "Color2": crater_color.outputs["Color"],
                },
            )

            # Create Principled BSDF
            principled_bsdf = nw.new_node(
                Nodes.PrincipledBSDF,
                input_kwargs={
                    "Base Color": final_color,
                    "Roughness": uniform(0.8, 0.9),  # Very rough for moon
                    "Metallic": 0.0,
                },
            )

            return principled_bsdf

    def add_orbital_animation(self, moon):
        """Add orbital motion around parent planet"""
        # Set orbital keyframes
        moon.location = (self.orbit_radius, 0, 0)
        moon.keyframe_insert(data_path="location", frame=1)

        # Orbit 360 degrees over 200 frames
        for frame in range(1, 201, 10):
            angle = (frame - 1) * 2 * math.pi / 200
            x = self.orbit_radius * math.cos(angle)
            y = self.orbit_radius * math.sin(angle)
            moon.location = (x, y, 0)
            moon.keyframe_insert(data_path="location", frame=frame)

        # Set interpolation to linear for smooth orbital motion
        if moon.animation_data and moon.animation_data.action:
            for fcurve in moon.animation_data.action.fcurves:
                if fcurve.data_path == "location":
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = "LINEAR"

    def add_rotation_animation(self, moon):
        """Add rotation animation to moon"""
        # Set rotation keyframes
        moon.rotation_euler = (0, 0, 0)
        moon.keyframe_insert(data_path="rotation_euler", frame=1)

        # Rotate 360 degrees over 150 frames
        moon.rotation_euler = (0, 0, 2 * math.pi)
        moon.keyframe_insert(data_path="rotation_euler", frame=150)

        # Set interpolation to linear for constant rotation
        if moon.animation_data and moon.animation_data.action:
            for fcurve in moon.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"
                    keyframe.interpolation = "LINEAR"
