# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Planets for Infinigen

import bpy
import gin
import numpy as np
from mathutils import Vector
from numpy.random import normal, uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import SurfaceTypes


class PlanetFactory(AssetFactory):
    """Base class for planet generation"""

    def __init__(self, factory_seed, radius=1.0, resolution=64, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.radius = radius
        self.resolution = resolution

    def create_asset(self, **kwargs):
        """Create planet asset with atmosphere and rings"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=self.radius, location=(0, 0, 0))
        planet = bpy.context.active_object
        planet.name = f"Planet_{self.factory_seed}"

        # Add geometry nodes for planet surface
        surface.add_geomod(planet, self.create_planet_geometry, apply=True)

        # Apply materials
        if hasattr(self, "material_func"):
            material = surface.shaderfunc_to_material(self.material_func)
            planet.active_material = material

        # Add atmosphere for gas giants
        if hasattr(self, "has_atmosphere") and self.has_atmosphere:
            self.add_atmosphere(planet)

        # Add rings for ringed planets
        if hasattr(self, "has_rings") and self.has_rings:
            self.add_rings(planet)

        # Add rotation animation
        self.add_rotation_animation(planet)

        tag_object(planet, SurfaceTypes.AstronomicalPlanet)
        return planet

    def add_atmosphere(self, planet):
        """Add atmospheric shell around planet"""
        # Create atmosphere sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=self.radius * 1.1, location=planet.location
        )
        atmosphere = bpy.context.active_object
        atmosphere.name = f"{planet.name}_Atmosphere"

        # Create atmospheric material
        mat = bpy.data.materials.new(name=f"{planet.name}_Atmosphere_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Add Principled BSDF
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (
            0.3,
            0.5,
            1.0,
            1.0,
        )  # Blue atmosphere
        bsdf.inputs["Alpha"].default_value = 0.1  # Very transparent
        bsdf.inputs["Roughness"].default_value = 0.0

        # Add Material Output
        output = nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Set material properties
        mat.blend_method = "BLEND"
        mat.show_transparent_back = False

        # Apply material
        atmosphere.data.materials.append(mat)

        # Parent to planet
        atmosphere.parent = planet

    def add_rings(self, planet):
        """Add ring system around planet"""
        # Create ring using torus
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.radius * 2.5,
            minor_radius=self.radius * 0.3,
            major_segments=64,
            minor_segments=16,
            location=planet.location,
        )
        rings = bpy.context.active_object
        rings.name = f"{planet.name}_Rings"

        # Create ring material
        mat = bpy.data.materials.new(name=f"{planet.name}_Rings_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Add Principled BSDF
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.8, 0.7, 0.5, 1.0)  # Golden rings
        bsdf.inputs["Alpha"].default_value = 0.8  # Semi-transparent
        bsdf.inputs["Roughness"].default_value = 0.2
        bsdf.inputs["Metallic"].default_value = 0.1

        # Add Material Output
        output = nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Set material properties
        mat.blend_method = "BLEND"

        # Apply material
        rings.data.materials.append(mat)

        # Parent to planet
        rings.parent = planet

    def add_rotation_animation(self, planet):
        """Add rotation animation to planet"""
        # Set rotation keyframes
        planet.rotation_euler = (0, 0, 0)
        planet.keyframe_insert(data_path="rotation_euler", frame=1)

        # Rotate 360 degrees over 100 frames
        planet.rotation_euler = (0, 0, 2 * 3.14159)  # 2π radians = 360 degrees
        planet.keyframe_insert(data_path="rotation_euler", frame=100)

        # Set interpolation to linear for constant rotation
        if planet.animation_data and planet.animation_data.action:
            for fcurve in planet.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


@gin.configurable
class RockyPlanetFactory(PlanetFactory):
    """Rocky planet like Earth, Mars, Venus"""

    def __init__(self, factory_seed, radius=1.0, resolution=64, coarse=False):
        super().__init__(factory_seed, radius, resolution, coarse)
        self.planet_type = "rocky"
        self.material_func = shader_rocky_planet

    def create_planet_geometry(self, nw: NodeWrangler):
        """Create rocky planet surface with craters and mountains"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create crater noise
        crater_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(5, 15),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.6, 0.8),
            },
        )

        # Create mountain ranges
        mountain_noise = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(2, 8),
                "Detail": uniform(5, 8),
                "Dimension": uniform(0.1, 0.3),
            },
        )

        # Combine noises for surface displacement
        surface_noise = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: crater_noise.outputs["Fac"],
                1: mountain_noise,
            },
            attrs={"operation": "ADD"},
        )

        # Apply displacement
        displacement = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: nw.new_node(Nodes.InputNormal),
                1: nw.new_value(0.1, "displacement_strength"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        set_position = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": group_input.outputs["Geometry"],
                "Offset": displacement.outputs["Vector"],
            },
        )

        # Custom Normals for enhanced planet surface detail (Blender 4.5+)
        # Temporarily disabled due to SetMeshNormal compatibility issues
        # if hasattr(Nodes, "SetMeshNormal"):
        #     normal = nw.new_node(Nodes.InputNormal)
        #
        #     # Create custom normal calculation for rocky planet surface
        #     planet_normal = nw.new_node(
        #         Nodes.VectorMath,
        #         input_kwargs={
        #             0: normal,
        #             1: nw.new_value(0.05, "rocky_normal_strength"),
        #         },
        #         attrs={"operation": "MULTIPLY"},
        #     )
        #
        #     # Apply custom normals to geometry
        #     set_normal = nw.new_node(
        #         Nodes.SetMeshNormal,
        #         input_kwargs={
        #             "Geometry": set_position,
        #             "Normal": planet_normal.outputs["Vector"],
        #         },
        #     )
        #
        #     return set_normal

        return set_position


def shader_rocky_planet(nw: NodeWrangler, random_seed=0):
    """Create shader for rocky planets like Earth, Mars"""
    nw.force_input_consistency()

    with FixedSeed(random_seed):
        # Use geometry input for shader context
        position = nw.new_node("ShaderNodeNewGeometry")

        # Create layered surface noise for more realistic terrain
        base_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(2, 5),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.6, 0.8),
            },
        )

        # Add crater noise
        crater_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(8, 20),
                "Detail": uniform(10, 15),
                "Roughness": uniform(0.7, 0.9),
            },
        )

        # Create mountain noise
        mountain_noise = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(3, 8),
                "Detail": uniform(5, 8),
                "Dimension": uniform(0.1, 0.3),
            },
        )

        # Combine noises for surface variation
        surface_mix = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: base_noise.outputs["Fac"],
                1: crater_noise.outputs["Fac"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        surface_final = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: surface_mix,
                1: mountain_noise,
            },
            attrs={"operation": "ADD"},
        )

        # Create realistic rock colors
        rock_color = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": surface_final})
        rock_color.color_ramp.elements[0].color = (
            uniform(0.3, 0.5),  # Red - darker rock
            uniform(0.2, 0.4),  # Green
            uniform(0.1, 0.3),  # Blue
            1.0,
        )
        rock_color.color_ramp.elements[1].color = (
            uniform(0.5, 0.7),  # Red - lighter rock
            uniform(0.4, 0.6),  # Green
            uniform(0.3, 0.5),  # Blue
            1.0,
        )

        # Add dust/oxide layer
        dust_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(15, 30),
                "Detail": uniform(12, 18),
            },
        )

        dust_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": dust_noise.outputs["Fac"]}
        )
        dust_color.color_ramp.elements[0].color = (
            uniform(0.4, 0.6),  # Red dust
            uniform(0.3, 0.5),  # Green
            uniform(0.2, 0.4),  # Blue
            1.0,
        )
        dust_color.color_ramp.elements[1].color = (
            uniform(0.2, 0.4),  # Darker dust
            uniform(0.1, 0.3),  # Green
            uniform(0.1, 0.2),  # Blue
            1.0,
        )

        # Mix rock and dust colors
        final_color = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": dust_noise.outputs["Fac"],
                "Color1": rock_color.outputs["Color"],
                "Color2": dust_color.outputs["Color"],
            },
        )

        # Create realistic roughness variation
        roughness = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": surface_final})
        roughness.color_ramp.elements[0].color = (
            uniform(0.7, 0.9),  # Rough areas
            uniform(0.7, 0.9),
            uniform(0.7, 0.9),
            1.0,
        )
        roughness.color_ramp.elements[1].color = (
            uniform(0.4, 0.6),  # Smooth areas
            uniform(0.4, 0.6),
            uniform(0.4, 0.6),
            1.0,
        )

        # Create Principled BSDF with enhanced properties
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": final_color,
                "Roughness": roughness.outputs["Color"],
                "Metallic": uniform(0.0, 0.1),  # Low metallic for rocky surface
            },
        )

        return principled_bsdf


@gin.configurable
class GasGiantFactory(PlanetFactory):
    """Gas giant like Jupiter, Saturn"""

    def __init__(
        self,
        factory_seed,
        radius=2.0,
        resolution=64,
        coarse=False,
        has_atmosphere=True,
        has_rings=False,
    ):
        super().__init__(factory_seed, radius, resolution, coarse)
        self.planet_type = "gas_giant"
        self.material_func = shader_gas_giant
        self.has_atmosphere = has_atmosphere
        self.has_rings = has_rings

    def create_planet_geometry(self, nw: NodeWrangler):
        """Create gas giant surface with atmospheric bands"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create atmospheric bands
        band_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 2.0),
                "Detail": uniform(3, 6),
            },
        )

        # Create storm systems
        storm_noise = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(1, 3),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Combine for atmospheric effects
        atmosphere = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: band_noise.outputs["Fac"],
                1: storm_noise.outputs["Distance"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Apply subtle displacement for atmospheric effects
        displacement = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: nw.new_node(Nodes.InputNormal),
                1: nw.new_value(0.02, "atmosphere_displacement"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        set_position = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": group_input.outputs["Geometry"],
                "Offset": displacement.outputs["Vector"],
            },
        )

        # Custom Normals for enhanced gas giant surface detail (Blender 4.5+)
        # Temporarily disabled due to SetMeshNormal compatibility issues
        # if hasattr(Nodes, "SetMeshNormal"):
        #     normal = nw.new_node(Nodes.InputNormal)
        #
        #     # Create custom normal calculation for gas giant surface
        #     planet_normal = nw.new_node(
        #         Nodes.VectorMath,
        #         input_kwargs={
        #             0: normal,
        #             1: nw.new_value(0.02, "gas_giant_normal_strength"),
        #         },
        #         attrs={"operation": "MULTIPLY"},
        #     )
        #
        #     # Apply custom normals to geometry
        #     set_normal = nw.new_node(
        #         Nodes.SetMeshNormal,
        #         input_kwargs={
        #             "Geometry": set_position,
        #             "Normal": planet_normal.outputs["Vector"],
        #         },
        #     )
        #
        #     return set_normal

        return set_position


def shader_gas_giant(nw: NodeWrangler, random_seed=0):
    """Create shader for gas giants like Jupiter, Saturn"""
    nw.force_input_consistency()

    with FixedSeed(random_seed):
        position = nw.new_node("ShaderNodeNewGeometry")

        # Create atmospheric bands with multiple layers
        band_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.3, 1.5),
                "Detail": uniform(4, 8),
                "Roughness": uniform(0.4, 0.7),
            },
        )

        # Add secondary band layer
        band_noise2 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(1.5, 4.0),
                "Detail": uniform(2, 5),
                "Roughness": uniform(0.6, 0.9),
            },
        )

        # Create storm systems with different scales
        storm_noise = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(0.5, 2.0),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Add smaller storm cells
        storm_cells = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(3, 8),
            },
            attrs={"voronoi_dimensions": "4D"},
        )

        # Create realistic gas giant colors (Jupiter-like)
        gas_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": band_noise.outputs["Fac"]}
        )
        gas_color.color_ramp.elements[0].color = (
            uniform(0.4, 0.7),  # Red - darker bands
            uniform(0.3, 0.5),  # Green
            uniform(0.2, 0.4),  # Blue
            1.0,
        )
        gas_color.color_ramp.elements[1].color = (
            uniform(0.7, 0.9),  # Red - lighter bands
            uniform(0.5, 0.7),  # Green
            uniform(0.4, 0.6),  # Blue
            1.0,
        )

        # Add secondary color variation
        gas_color2 = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": band_noise2.outputs["Fac"]}
        )
        gas_color2.color_ramp.elements[0].color = (
            uniform(0.5, 0.8),  # Red
            uniform(0.4, 0.6),  # Green
            uniform(0.3, 0.5),  # Blue
            1.0,
        )
        gas_color2.color_ramp.elements[1].color = (
            uniform(0.3, 0.5),  # Red - darker
            uniform(0.2, 0.4),  # Green
            uniform(0.1, 0.3),  # Blue
            1.0,
        )

        # Mix band colors
        band_mix = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": band_noise2.outputs["Fac"],
                "Color1": gas_color.outputs["Color"],
                "Color2": gas_color2.outputs["Color"],
            },
        )

        # Create storm effects
        storm_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": storm_noise.outputs["Distance"]}
        )
        storm_color.color_ramp.elements[0].color = (0.1, 0.05, 0.1, 1.0)  # Dark storm
        storm_color.color_ramp.elements[1].color = (0.9, 0.8, 0.7, 1.0)  # Light areas

        # Add storm cell effects
        storm_cell_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": storm_cells.outputs["Distance"]}
        )
        storm_cell_color.color_ramp.elements[0].color = (
            0.2,
            0.1,
            0.1,
            1.0,
        )  # Dark cells
        storm_cell_color.color_ramp.elements[1].color = (
            0.8,
            0.7,
            0.6,
            1.0,
        )  # Light areas

        # Mix storm effects
        storm_mix = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": storm_cells.outputs["Distance"],
                "Color1": storm_color.outputs["Color"],
                "Color2": storm_cell_color.outputs["Color"],
            },
        )

        # Final color mixing
        final_color = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": storm_mix,
                "Color1": band_mix,
                "Color2": storm_mix,
            },
        )

        # Create atmospheric glow effect
        emission_strength = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": band_noise.outputs["Fac"]}
        )
        emission_strength.color_ramp.elements[0].color = (
            0.0,
            0.0,
            0.0,
            1.0,
        )  # No emission
        emission_strength.color_ramp.elements[1].color = (
            0.1,
            0.1,
            0.1,
            1.0,
        )  # Subtle emission

        # Create Principled BSDF with atmospheric properties
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": final_color,
                "Roughness": uniform(0.9, 1.0),  # Very rough for gas
                "Metallic": 0.0,
            },
        )

        return principled_bsdf


@gin.configurable
class RingedPlanetFactory(GasGiantFactory):
    """Ringed gas giant like Saturn"""

    def __init__(self, factory_seed, radius=2.0, resolution=64, coarse=False):
        super().__init__(
            factory_seed,
            radius,
            resolution,
            coarse,
            has_atmosphere=True,
            has_rings=True,
        )
        self.planet_type = "ringed_gas_giant"


@gin.configurable
class IcePlanetFactory(PlanetFactory):
    """Ice planet like Europa, Enceladus"""

    def __init__(self, factory_seed, radius=0.8, resolution=64, coarse=False):
        super().__init__(factory_seed, radius, resolution, coarse)
        self.planet_type = "ice"
        self.material_func = shader_ice_planet

    def create_planet_geometry(self, nw: NodeWrangler):
        """Create ice planet surface with cracks and ridges"""
        group_input = nw.new_node(Nodes.GroupInput)

        # For now, just return the base geometry
        # TODO: Implement proper ice surface effects
        return group_input.outputs["Geometry"]


def shader_ice_planet(nw: NodeWrangler, random_seed=0):
    """Create shader for ice planets like Europa, Enceladus with enhanced cracks and reflections"""
    nw.force_input_consistency()

    with FixedSeed(random_seed):
        position = nw.new_node("ShaderNodeNewGeometry")

        # Create primary ice surface noise
        ice_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(8, 15),
                "Detail": uniform(12, 18),
                "Roughness": uniform(0.6, 0.8),
            },
        )

        # Create secondary ice variation
        ice_variation = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(20, 30),
                "Detail": uniform(8, 12),
                "Roughness": uniform(0.5, 0.7),
            },
            attrs={"musgrave_dimensions": "3D"},
        )

        # Create large crack patterns using Voronoi
        crack_noise = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(2, 4),
                "Randomness": uniform(0.8, 0.95),
            },
            attrs={"voronoi_dimensions": "2D"},
        )

        # Create fine crack details
        fine_cracks = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(10, 15),
                "Randomness": uniform(0.6, 0.8),
            },
            attrs={"voronoi_dimensions": "2D"},
        )

        # Combine ice noises for more realistic surface
        combined_ice = nw.new_node(
            Nodes.Math,
            input_kwargs={
                "Value": ice_noise.outputs["Fac"],
                "Value_001": ice_variation.outputs["Fac"],
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create ice colors with more realistic blue-white tones
        ice_color = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": combined_ice})
        ice_color.color_ramp.elements[0].color = (
            uniform(0.85, 0.95),  # Very light blue-white
            uniform(0.9, 1.0),  # Light cyan
            uniform(0.95, 1.0),  # Almost white
            1.0,
        )
        ice_color.color_ramp.elements[1].color = (
            uniform(0.6, 0.8),  # Medium blue
            uniform(0.7, 0.9),  # Light blue
            uniform(0.8, 1.0),  # Light cyan
            1.0,
        )

        # Create large crack color (very dark for contrast)
        crack_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": crack_noise.outputs["Distance"]}
        )
        crack_color.color_ramp.elements[0].color = (
            0.05,  # Very dark blue-black
            0.08,
            0.12,
            1.0,
        )
        crack_color.color_ramp.elements[1].color = (
            0.2,  # Dark blue
            0.25,
            0.3,
            1.0,
        )

        # Create fine crack color
        fine_crack_color = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": fine_cracks.outputs["Distance"]}
        )
        fine_crack_color.color_ramp.elements[0].color = (
            0.1,  # Dark blue
            0.15,
            0.2,
            1.0,
        )
        fine_crack_color.color_ramp.elements[1].color = (
            0.4,  # Medium blue
            0.5,
            0.6,
            1.0,
        )

        # Mix ice and large crack colors
        ice_crack_mix = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": crack_noise.outputs["Distance"],
                "Color1": ice_color.outputs["Color"],
                "Color2": crack_color.outputs["Color"],
            },
        )

        # Add fine cracks
        final_color = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": fine_cracks.outputs["Distance"],
                "Color1": ice_crack_mix,
                "Color2": fine_crack_color.outputs["Color"],
            },
        )

        # Create realistic roughness for ice with variation
        roughness = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": combined_ice})
        roughness.color_ramp.elements[0].color = (
            0.05,  # Very smooth ice
            0.05,
            0.05,
            1.0,
        )
        roughness.color_ramp.elements[1].color = (
            0.2,  # Slightly rough areas
            0.2,
            0.2,
            1.0,
        )

        # Create crack roughness
        crack_roughness = nw.new_node(
            Nodes.ColorRamp, input_kwargs={"Fac": crack_noise.outputs["Distance"]}
        )
        crack_roughness.color_ramp.elements[0].color = (
            0.8,  # Very rough cracks
            0.8,
            0.8,
            1.0,
        )
        crack_roughness.color_ramp.elements[1].color = (
            0.1,  # Smooth areas
            0.1,
            0.1,
            1.0,
        )

        # Mix roughness values
        final_roughness = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                "Fac": crack_noise.outputs["Distance"],
                "Color1": roughness.outputs["Color"],
                "Color2": crack_roughness.outputs["Color"],
            },
        )

        # Create metallic variation for ice reflections
        metallic = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": combined_ice})
        metallic.color_ramp.elements[0].color = (
            0.0,  # Non-metallic
            0.0,
            0.0,
            1.0,
        )
        metallic.color_ramp.elements[1].color = (
            0.1,  # Slightly metallic for ice
            0.1,
            0.1,
            1.0,
        )

        # Create Principled BSDF for ice planet with enhanced properties
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": final_color,
                "Roughness": final_roughness,
                "Metallic": metallic,
            },
        )

        return principled_bsdf
