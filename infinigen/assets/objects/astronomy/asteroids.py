# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Asteroids and Comets for Infinigen

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


@gin.configurable
class AsteroidBeltFactory(AssetFactory):
    """Generate asteroid belts and individual asteroids"""

    def __init__(self, factory_seed, asteroid_count=100, radius=20.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.asteroid_count = asteroid_count
        self.radius = radius

    def create_asteroid_belt_geometry(self, nw: NodeWrangler):
        """Create asteroid belt with procedural asteroids"""
        group_input = nw.new_node(Nodes.GroupInput)

        # For now, just return the base geometry
        # TODO: Implement proper asteroid belt distribution
        return group_input.outputs["Geometry"]

    def create_asteroid_geometry(self, nw: NodeWrangler):
        """Create individual asteroid geometry"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create irregular asteroid shape using noise
        deformation_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(2, 8),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.7, 0.9),
            },
        )

        # Create additional surface detail
        detail_noise = nw.new_node(
            Nodes.MusgraveTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(5, 15),
                "Detail": uniform(8, 12),
                "Dimension": uniform(0.1, 0.4),
            },
        )

        # Combine noises for irregular surface
        surface_deformation = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: deformation_noise.outputs["Fac"],
                1: detail_noise,
            },
            attrs={"operation": "ADD"},
        )

        # Apply deformation to create irregular shape
        displacement = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: nw.new_node(Nodes.InputNormal),
                1: nw.new_value(0.2, "asteroid_deformation"),
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

        # Custom Normals for enhanced asteroid surface detail (Blender 4.5+)
        # Temporarily disabled due to SetMeshNormal compatibility issues
        # if hasattr(Nodes, "SetMeshNormal"):
        #     normal = nw.new_node(Nodes.InputNormal)
        #
        #     # Create custom normal calculation for asteroid surface
        #     asteroid_normal = nw.new_node(
        #         Nodes.VectorMath,
        #         input_kwargs={
        #             0: normal,
        #             1: nw.new_value(0.08, "asteroid_normal_strength"),
        #         },
        #         attrs={"operation": "MULTIPLY"},
        #     )
        #
        #     # Apply custom normals to geometry
        #     set_normal = nw.new_node(
        #         Nodes.SetMeshNormal,
        #         input_kwargs={
        #             "Geometry": set_position,
        #             "Normal": asteroid_normal.outputs["Vector"],
        #         },
        #     )
        #
        #     return set_normal

        return set_position

    def create_asset(self, **kwargs):
        """Create asteroid belt asset"""
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.radius,
            minor_radius=2.0,
            major_segments=32,
            minor_segments=16,
        )
        asteroid_belt = bpy.context.active_object
        asteroid_belt.name = f"AsteroidBelt_{self.factory_seed}"

        # Add geometry nodes for asteroid belt
        surface.add_geomod(
            asteroid_belt, self.create_asteroid_belt_geometry, apply=True
        )

        # Add rotation animation
        self.add_rotation_animation(asteroid_belt)

        # Tag object
        tag_object(asteroid_belt, SurfaceTypes.SDFPerturb)

        return asteroid_belt

    def add_rotation_animation(self, asteroid_belt):
        """Add rotation animation to asteroid belt"""
        # Set rotation keyframes
        asteroid_belt.rotation_euler = (0, 0, 0)
        asteroid_belt.keyframe_insert(data_path="rotation_euler", frame=1)

        # Rotate 360 degrees over 200 frames (slower than planets)
        asteroid_belt.rotation_euler = (0, 0, 2 * 3.14159)  # 2π radians = 360 degrees
        asteroid_belt.keyframe_insert(data_path="rotation_euler", frame=200)

        # Set interpolation to linear for constant rotation
        if asteroid_belt.animation_data and asteroid_belt.animation_data.action:
            for fcurve in asteroid_belt.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


@gin.configurable
class CometFactory(AssetFactory):
    """Generate comets with tails"""

    def __init__(self, factory_seed, tail_length=10.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.tail_length = tail_length

    def create_comet_geometry(self, nw: NodeWrangler):
        """Create comet with nucleus and tail"""
        group_input = nw.new_node(Nodes.GroupInput)
        position = nw.new_node(Nodes.InputPosition)

        # Create comet nucleus with irregular shape
        nucleus_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(3, 8),
                "Detail": uniform(6, 10),
                "Roughness": uniform(0.6, 0.8),
            },
        )

        # Apply nucleus deformation
        nucleus_displacement = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: nw.new_node(Nodes.InputNormal),
                1: nw.new_value(0.15, "nucleus_deformation"),
            },
            attrs={"operation": "MULTIPLY"},
        )

        # Create tail geometry using curve
        tail_curve = nw.new_node(
            Nodes.CurveLine,
            input_kwargs={
                "Start": (0, 0, 0),
                "End": (0, 0, -self.tail_length),
            },
        )

        # Add tail variation
        tail_noise = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                "Vector": position,
                "Scale": uniform(1, 3),
                "Detail": uniform(4, 8),
            },
        )

        # Create tail mesh
        tail_geometry = nw.new_node(
            Nodes.CurveToMesh,
            input_kwargs={
                "Curve": tail_curve,
                "Profile Curve": nw.new_node(
                    Nodes.MeshCircle, input_kwargs={"Radius": uniform(0.1, 0.3)}
                ),
            },
        )

        # Apply nucleus displacement
        nucleus_geometry = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": group_input.outputs["Geometry"],
                "Offset": nucleus_displacement.outputs["Vector"],
            },
        )

        # Join nucleus and tail
        comet_geometry = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={"Geometry": [nucleus_geometry, tail_geometry]},
        )

        # Custom Normals for enhanced comet surface detail (Blender 4.5+)
        # Temporarily disabled due to SetMeshNormal compatibility issues
        # if hasattr(Nodes, "SetMeshNormal"):
        #     normal = nw.new_node(Nodes.InputNormal)
        #
        #     # Create custom normal calculation for comet surface
        #     comet_normal = nw.new_node(
        #         Nodes.VectorMath,
        #         input_kwargs={
        #             0: normal,
        #             1: nw.new_value(0.03, "comet_normal_strength"),
        #         },
        #         attrs={"operation": "MULTIPLY"},
        #     )
        #
        #     # Apply custom normals to geometry
        #     set_normal = nw.new_node(
        #         Nodes.SetMeshNormal,
        #         input_kwargs={
        #             "Geometry": comet_geometry,
        #             "Normal": comet_normal.outputs["Vector"],
        #         },
        #     )
        #
        #     return set_normal

        return comet_geometry

    def create_asset(self, **kwargs):
        """Create comet asset"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0))
        comet = bpy.context.active_object
        comet.name = f"Comet_{self.factory_seed}"

        # Add geometry nodes for comet
        surface.add_geomod(comet, self.create_comet_geometry, apply=True)

        # Add movement animation
        self.add_movement_animation(comet)

        # Tag object
        tag_object(comet, SurfaceTypes.SDFPerturb)

        return comet

    def add_movement_animation(self, comet):
        """Add movement animation to comet"""
        # Set initial position
        comet.location = (0, 0, 0)
        comet.keyframe_insert(data_path="location", frame=1)

        # Move comet in elliptical orbit
        comet.location = (20, 0, 0)
        comet.keyframe_insert(data_path="location", frame=50)

        comet.location = (0, 20, 0)
        comet.keyframe_insert(data_path="location", frame=100)

        # Add rotation
        comet.rotation_euler = (0, 0, 0)
        comet.keyframe_insert(data_path="rotation_euler", frame=1)

        comet.rotation_euler = (0, 0, 4 * 3.14159)  # 2 full rotations
        comet.keyframe_insert(data_path="rotation_euler", frame=100)

        # Set interpolation to linear for smooth movement
        if comet.animation_data and comet.animation_data.action:
            for fcurve in comet.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


@gin.configurable
class MeteorFactory(AssetFactory):
    """Generate meteors and meteorites"""

    def __init__(self, factory_seed, meteor_size=0.2, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.meteor_size = meteor_size

    def create_meteor_geometry(self, nw: NodeWrangler):
        """Create meteor geometry"""
        group_input = nw.new_node(Nodes.GroupInput)

        # For now, just return the base geometry
        # TODO: Implement proper meteor geometry
        return group_input.outputs["Geometry"]

    def create_asset(self, **kwargs):
        """Create meteor asset"""
        bpy.ops.mesh.primitive_cone_add(
            radius1=self.meteor_size, depth=self.meteor_size * 2, location=(0, 0, 0)
        )
        meteor = bpy.context.active_object
        meteor.name = f"Meteor_{self.factory_seed}"

        # Add geometry nodes for meteor
        surface.add_geomod(meteor, self.create_meteor_geometry, apply=True)

        # Tag object
        tag_object(meteor, SurfaceTypes.SDFPerturb)

        return meteor
