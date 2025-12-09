# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Modern Tree Factory with Addon Integration

import logging

import bpy
import gin
import numpy as np

from infinigen.core.init import require_blender_addon
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


@gin.configurable
class TreeFactory(AssetFactory):
    """Modern Tree Factory with Mtree and TreeGen addon integration"""

    scale = 1.0  # Trees in meters

    def __init__(
        self,
        factory_seed,
        tree_type="oak",
        addon_type="simple",  # "simple" (default), "mtree" or "treegen"
        height_range=(3.0, 8.0),
        trunk_radius_range=(0.2, 0.6),
        crown_radius_range=(2.0, 5.0),
        season="summer",
        coarse=False,
        **kwargs,
    ):
        super(TreeFactory, self).__init__(factory_seed, coarse=coarse)

        self.tree_type = tree_type
        self.addon_type = addon_type
        self.height_range = height_range
        self.trunk_radius_range = trunk_radius_range
        self.crown_radius_range = crown_radius_range
        self.season = season

        # Ensure required addon is available
        self._ensure_addon_available()

        logger.debug(f"TreeFactory initialized: {tree_type}, {addon_type}")

    def _ensure_addon_available(self):
        """Ensure the required tree addon is available"""
        try:
            if self.addon_type == "mtree":
                # Mtree deaktiviert wegen Endlosschleife in Blender 4.5
                logger.warning("Mtree addon disabled due to infinite loop issue, using simple tree generation")
                self.addon_type = "simple"
            elif self.addon_type == "treegen":
                require_blender_addon("treegen", fail="warn")
        except Exception as e:
            logger.warning(f"Tree addon {self.addon_type} not available: {e}")
            # Fallback to simple tree generation
            self.addon_type = "simple"

    def create_placeholder(self, i, loc, rot):
        """Create improved tree placeholder with better geometry"""
        logger.debug(f"Creating improved tree placeholder {i}")

        # Create a more realistic tree shape using multiple objects
        tree_group = bpy.data.collections.new(f"Tree_{i}")
        bpy.context.scene.collection.children.link(tree_group)

        # Create trunk
        trunk = butil.spawn_cube(
            size=1, location=(loc[0], loc[1], loc[2] + 1), name=f"Trunk_{i}"
        )
        trunk.scale = (0.3, 0.3, 2.0)  # Make it tall and thin
        trunk.rotation_euler = rot
        trunk.data.materials.append(self._create_bark_material())
        tree_group.objects.link(trunk)

        # Create crown (leaves)
        crown = butil.spawn_cube(
            size=3, location=(loc[0], loc[1], loc[2] + 3), name=f"Crown_{i}"
        )
        crown.scale = (1.5, 1.5, 1.0)  # Make it wide and flat
        crown.rotation_euler = rot
        crown.data.materials.append(self._create_leaves_material())
        tree_group.objects.link(crown)  # Fix: link crown, not trunk again

        # Create main tree object
        placeholder = butil.spawn_cube(
            size=4, location=loc, name=f"Tree_Placeholder_{i}"
        )
        placeholder.rotation_euler = rot
        placeholder.data.materials.append(self._create_tree_material())

        # Tag for Infinigen compatibility
        placeholder["is_placeholder"] = True
        placeholder["tree_type"] = self.tree_type
        placeholder["addon_type"] = self.addon_type
        placeholder["coarse_mesh_placeholder"] = self.coarse
        placeholder["tree_group"] = tree_group.name

        return placeholder

    def _create_bark_material(self):
        """Create bark material for tree trunk"""
        material = bpy.data.materials.new(name="BarkMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add Principled BSDF
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.inputs["Base Color"].default_value = (0.4, 0.2, 0.1, 1.0)  # Brown
        principled.inputs["Roughness"].default_value = 0.8
        principled.inputs["Metallic"].default_value = 0.0

        # Add Material Output
        output = nodes.new(type="ShaderNodeOutputMaterial")

        # Connect nodes
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        return material

    def _create_leaves_material(self):
        """Create leaves material for tree crown"""
        material = bpy.data.materials.new(name="LeavesMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add Principled BSDF
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.inputs["Base Color"].default_value = (0.2, 0.6, 0.2, 1.0)  # Green
        principled.inputs["Roughness"].default_value = 0.7
        principled.inputs["Metallic"].default_value = 0.0

        # Add Material Output
        output = nodes.new(type="ShaderNodeOutputMaterial")

        # Connect nodes
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        return material

    def _create_tree_material(self):
        """Create main tree material"""
        material = bpy.data.materials.new(name="TreeMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add Principled BSDF
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.inputs["Base Color"].default_value = (0.3, 0.5, 0.2, 1.0)  # Green-brown
        principled.inputs["Roughness"].default_value = 0.6
        principled.inputs["Metallic"].default_value = 0.0

        # Add Material Output
        output = nodes.new(type="ShaderNodeOutputMaterial")

        # Connect nodes
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        return material

    def create_asset(self, placeholder, face_size, distance, **kwargs):
        """Create actual tree asset using modern addons"""
        logger.debug(f"Creating tree asset: {self.tree_type}")

        try:
            if self.addon_type == "mtree":
                return self._create_mtree_asset(placeholder, face_size, distance)
            elif self.addon_type == "treegen":
                return self._create_treegen_asset(placeholder, face_size, distance)
            else:
                return self._create_simple_tree_asset(placeholder, face_size, distance)

        except Exception as e:
            logger.error(f"Error creating tree asset: {e}")
            # Fallback to simple tree
            return self._create_simple_tree_asset(placeholder, face_size, distance)

    def _create_mtree_asset(self, placeholder, face_size, distance):
        """Create tree using Mtree addon"""
        try:
            # Use Mtree addon to generate tree
            bpy.ops.object.select_all(action="DESELECT")
            placeholder.select_set(True)
            bpy.context.view_layer.objects.active = placeholder

            # Generate random parameters
            height = np.random.uniform(*self.height_range)
            trunk_radius = np.random.uniform(*self.trunk_radius_range)
            crown_radius = np.random.uniform(*self.crown_radius_range)

            # Call Mtree addon (this would need to be adapted to actual Mtree API)
            # bpy.ops.mtree.generate_tree(
            #     height=height,
            #     trunk_radius=trunk_radius,
            #     crown_radius=crown_radius,
            #     tree_type=self.tree_type
            # )

            # For now, create a simple tree as placeholder
            tree_obj = self._create_simple_tree_asset(placeholder, face_size, distance)
            tree_obj.name = f"Mtree_{self.tree_type}_{self.factory_seed}"

            return tree_obj

        except Exception as e:
            logger.warning(f"Mtree generation failed: {e}")
            return self._create_simple_tree_asset(placeholder, face_size, distance)

    def _create_treegen_asset(self, placeholder, face_size, distance):
        """Create tree using TreeGen addon"""
        try:
            # Use TreeGen addon to generate tree
            bpy.ops.object.select_all(action="DESELECT")
            placeholder.select_set(True)
            bpy.context.view_layer.objects.active = placeholder

            # Generate random parameters
            height = np.random.uniform(*self.height_range)
            trunk_radius = np.random.uniform(*self.trunk_radius_range)
            crown_radius = np.random.uniform(*self.crown_radius_range)

            # Call TreeGen addon (this would need to be adapted to actual TreeGen API)
            # bpy.ops.treegen.generate_tree(
            #     height=height,
            #     trunk_radius=trunk_radius,
            #     crown_radius=crown_radius,
            #     tree_type=self.tree_type
            # )

            # For now, create a simple tree as placeholder
            tree_obj = self._create_simple_tree_asset(placeholder, face_size, distance)
            tree_obj.name = f"TreeGen_{self.tree_type}_{self.factory_seed}"

            return tree_obj

        except Exception as e:
            logger.warning(f"TreeGen generation failed: {e}")
            return self._create_simple_tree_asset(placeholder, face_size, distance)

    def _create_simple_tree_asset(self, placeholder, face_size, distance):
        """Create simple tree asset as fallback"""
        # Generate random parameters
        height = np.random.uniform(*self.height_range)
        trunk_radius = np.random.uniform(*self.trunk_radius_range)
        crown_radius = np.random.uniform(*self.crown_radius_range)

        # Create trunk
        bpy.ops.mesh.primitive_cylinder_add(
            radius=trunk_radius, depth=height, location=placeholder.location
        )
        trunk = bpy.context.active_object
        trunk.name = f"Trunk_{self.tree_type}_{self.factory_seed}"

        # Create crown
        crown_location = (
            placeholder.location[0],
            placeholder.location[1],
            placeholder.location[2] + height / 2,
        )
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=crown_radius, location=crown_location
        )
        crown = bpy.context.active_object
        crown.name = f"Crown_{self.tree_type}_{self.factory_seed}"

        # Scale crown to be more elliptical
        crown.scale = (1.0, 1.0, 0.7)

        # Join trunk and crown
        bpy.ops.object.select_all(action="DESELECT")
        trunk.select_set(True)
        crown.select_set(True)
        bpy.context.view_layer.objects.active = trunk
        bpy.ops.object.join()

        tree_obj = bpy.context.active_object
        tree_obj.name = f"Simple_{self.tree_type}_{self.factory_seed}"

        # Apply materials based on season
        self._apply_tree_material(tree_obj)

        # Tag for Infinigen compatibility
        tag_object(tree_obj, "tree")

        return tree_obj

    def _apply_tree_material(self, obj):
        """Apply appropriate material based on tree type and season"""
        try:
            # Create material
            material = bpy.data.materials.new(
                name=f"{self.tree_type}_{self.season}_Material"
            )
            material.use_nodes = True

            # Configure Principled BSDF
            bsdf = material.node_tree.nodes["Principled BSDF"]

            # Set colors based on tree type and season
            if self.tree_type == "oak":
                if self.season == "autumn":
                    bsdf.inputs["Base Color"].default_value = (
                        0.8,
                        0.4,
                        0.1,
                        1.0,
                    )  # Orange
                else:
                    bsdf.inputs["Base Color"].default_value = (
                        0.2,
                        0.6,
                        0.2,
                        1.0,
                    )  # Green
            elif self.tree_type == "pine":
                bsdf.inputs["Base Color"].default_value = (
                    0.1,
                    0.4,
                    0.1,
                    1.0,
                )  # Dark green
            elif self.tree_type == "maple":
                if self.season == "autumn":
                    bsdf.inputs["Base Color"].default_value = (
                        0.9,
                        0.3,
                        0.1,
                        1.0,
                    )  # Red
                else:
                    bsdf.inputs["Base Color"].default_value = (
                        0.3,
                        0.7,
                        0.2,
                        1.0,
                    )  # Light green
            else:
                bsdf.inputs["Base Color"].default_value = (
                    0.2,
                    0.5,
                    0.2,
                    1.0,
                )  # Default green

            bsdf.inputs["Roughness"].default_value = 0.8

            # Add material to object
            obj.data.materials.append(material)

        except Exception as e:
            logger.warning(f"Could not apply tree material: {e}")


@gin.configurable
class BushFactory(TreeFactory):
    """Bush Factory - extends TreeFactory with bush parameters"""

    def __init__(
        self,
        factory_seed,
        tree_type="bush",
        addon_type="mtree",
        height_range=(0.5, 2.0),
        trunk_radius_range=(0.1, 0.3),
        crown_radius_range=(1.0, 3.0),
        season="summer",
        coarse=False,
        **kwargs,
    ):
        super(BushFactory, self).__init__(
            factory_seed=factory_seed,
            tree_type=tree_type,
            addon_type=addon_type,
            height_range=height_range,
            trunk_radius_range=trunk_radius_range,
            crown_radius_range=crown_radius_range,
            season=season,
            coarse=coarse,
            **kwargs,
        )

        # Override for bush characteristics
        self.height_range = height_range
        self.trunk_radius_range = trunk_radius_range
        self.crown_radius_range = crown_radius_range
