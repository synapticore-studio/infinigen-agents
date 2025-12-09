#!/usr/bin/env python3
"""
Simple Tree System for Infinigen
Uses Blender's built-in mesh creation following Infinigen patterns
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Euler
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


class SimpleTreeGenerator:
    """Simple tree generator using Blender's built-in meshes"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Tree types and parameters
        self.tree_types = {
            "oak": {
                "trunk_height": (3.0, 8.0),
                "trunk_radius": (0.3, 0.8),
                "crown_radius": (2.0, 5.0),
                "crown_height": (2.0, 4.0),
                "branch_count": (8, 15),
                "leaf_density": 0.7,
            },
            "pine": {
                "trunk_height": (5.0, 15.0),
                "trunk_radius": (0.2, 0.6),
                "crown_radius": (1.5, 3.0),
                "crown_height": (3.0, 8.0),
                "branch_count": (12, 20),
                "leaf_density": 0.9,
            },
            "maple": {
                "trunk_height": (4.0, 10.0),
                "trunk_radius": (0.4, 0.9),
                "crown_radius": (3.0, 6.0),
                "crown_height": (2.5, 5.0),
                "branch_count": (6, 12),
                "leaf_density": 0.8,
            },
            "palm": {
                "trunk_height": (6.0, 12.0),
                "trunk_radius": (0.3, 0.7),
                "crown_radius": (2.0, 4.0),
                "crown_height": (1.0, 2.0),
                "branch_count": (8, 15),
                "leaf_density": 0.6,
            },
            "bush": {
                "trunk_height": (0.5, 2.0),
                "trunk_radius": (0.1, 0.3),
                "crown_radius": (1.0, 3.0),
                "crown_height": (1.0, 2.5),
                "branch_count": (4, 8),
                "leaf_density": 0.9,
            },
        }

    def generate_tree(
        self,
        tree_type: str = "oak",
        seed: int = 42,
        position: Tuple[float, float, float] = (0, 0, 0),
        scale: float = 1.0,
    ) -> Optional[bpy.types.Object]:
        """Generate a simple tree"""

        if tree_type not in self.tree_types:
            tree_type = "oak"

        # Set seed for reproducible results
        np.random.seed(seed)

        try:
            # Tree parameters
            params = self.tree_types[tree_type]

            # Random parameters based on seed
            trunk_height = np.random.uniform(*params["trunk_height"]) * scale
            trunk_radius = np.random.uniform(*params["trunk_radius"]) * scale
            crown_radius = np.random.uniform(*params["crown_radius"]) * scale
            crown_height = np.random.uniform(*params["crown_height"]) * scale
            branch_count = np.random.randint(*params["branch_count"])
            leaf_density = params["leaf_density"]

            # Create tree
            tree_obj = self._create_simple_tree(
                tree_type,
                trunk_height,
                trunk_radius,
                crown_radius,
                crown_height,
                branch_count,
                leaf_density,
                position,
            )

            if tree_obj:
                tree_obj.name = f"{tree_type.title()}_Tree_{seed}"
                tree_obj["tree_type"] = tree_type
                tree_obj["tree_seed"] = seed
                tree_obj["generation_time"] = time.time()

                # Apply material
                self._apply_tree_material(tree_obj, tree_type)

                self.logger.info(f"✅ {tree_type.title()} tree generated: {tree_obj.name}")
                return tree_obj

        except Exception as e:
            self.logger.error(f"❌ Error generating tree: {e}")
            return None

    def _create_simple_tree(
        self,
        tree_type: str,
        trunk_height: float,
        trunk_radius: float,
        crown_radius: float,
        crown_height: float,
        branch_count: int,
        leaf_density: float,
        position: Tuple[float, float, float],
    ) -> Optional[bpy.types.Object]:
        """Create simple tree using Blender's built-in meshes"""

        try:
            # Create trunk
            bpy.ops.mesh.primitive_cylinder_add(
                radius=trunk_radius,
                depth=trunk_height,
                location=(position[0], position[1], position[2] + trunk_height / 2),
            )
            trunk = bpy.context.active_object
            trunk.name = f"{tree_type.title()}_Trunk"

            # Create crown
            if tree_type == "palm":
                # Palm - simplified crown
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=crown_radius,
                    location=(
                        position[0],
                        position[1],
                        position[2] + trunk_height + crown_height / 2,
                    ),
                )
                crown = bpy.context.active_object
                crown.name = f"{tree_type.title()}_Crown"

                # Scale crown (elliptical)
                crown.scale = (1.0, 1.0, crown_height / crown_radius)

            else:
                # Standard tree
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=crown_radius,
                    location=(
                        position[0],
                        position[1],
                        position[2] + trunk_height + crown_height / 2,
                    ),
                )
                crown = bpy.context.active_object
                crown.name = f"{tree_type.title()}_Crown"

                # Scale crown (elliptical)
                crown.scale = (1.0, 1.0, crown_height / crown_radius)

            # Add branches (simplified)
            for i in range(min(branch_count, 8)):  # Max 8 branches
                angle = (2 * np.pi * i) / min(branch_count, 8)
                branch_height = trunk_height * (0.3 + 0.4 * np.random.random())
                branch_length = crown_radius * (0.5 + 0.5 * np.random.random())

                # Branch position
                x = position[0] + np.cos(angle) * trunk_radius * 1.2
                y = position[1] + np.sin(angle) * trunk_radius * 1.2
                z = position[2] + branch_height

                # Create branch
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=trunk_radius * 0.3, depth=branch_length, location=(x, y, z)
                )
                branch = bpy.context.active_object
                branch.name = f"{tree_type.title()}_Branch_{i}"

                # Rotate branch
                rotation_angle = np.random.uniform(-30, 30)
                branch.rotation_euler = (0, 0, np.radians(rotation_angle))

            # Combine all parts into one object
            tree_obj = self._combine_tree_parts(tree_type, position)

            return tree_obj

        except Exception as e:
            self.logger.error(f"Error creating tree: {e}")
            return None

    def _combine_tree_parts(
        self, tree_type: str, position: Tuple[float, float, float]
    ) -> Optional[bpy.types.Object]:
        """Combine tree parts into one object"""

        try:
            # Collect all tree objects
            tree_parts = []
            for obj in bpy.context.scene.objects:
                if obj.name.startswith(f"{tree_type.title()}_"):
                    tree_parts.append(obj)

            if not tree_parts:
                return None

            # Choose first object as base
            tree_obj = tree_parts[0]
            tree_obj.name = f"{tree_type.title()}_Tree_Combined"

            # Add other parts
            for part in tree_parts[1:]:
                # Add modifier
                modifier = tree_obj.modifiers.new(name="Boolean", type="BOOLEAN")
                modifier.operation = "UNION"
                modifier.object = part

                # Apply boolean
                bpy.context.view_layer.objects.active = tree_obj
                bpy.ops.object.modifier_apply(modifier=modifier.name)

                # Delete part
                bpy.data.objects.remove(part, do_unlink=True)

            return tree_obj

        except Exception as e:
            self.logger.error(f"Error combining tree parts: {e}")
            return None

    def _apply_tree_material(self, obj: bpy.types.Object, tree_type: str):
        """Apply tree material"""

        try:
            # Create material
            material = bpy.data.materials.new(name=f"{tree_type.title()}_Material")
            material.use_nodes = True

            # Configure Principled BSDF
            bsdf = material.node_tree.nodes["Principled BSDF"]

            if tree_type == "oak":
                bsdf.inputs["Base Color"].default_value = (0.4, 0.3, 0.1, 1.0)  # Brown
                bsdf.inputs["Roughness"].default_value = 0.8
            elif tree_type == "pine":
                bsdf.inputs["Base Color"].default_value = (0.2, 0.4, 0.2, 1.0)  # Green
                bsdf.inputs["Roughness"].default_value = 0.9
            elif tree_type == "maple":
                bsdf.inputs["Base Color"].default_value = (0.6, 0.4, 0.2, 1.0)  # Orange-Brown
                bsdf.inputs["Roughness"].default_value = 0.7
            elif tree_type == "palm":
                bsdf.inputs["Base Color"].default_value = (0.3, 0.5, 0.2, 1.0)  # Green
                bsdf.inputs["Roughness"].default_value = 0.6
            else:  # bush
                bsdf.inputs["Base Color"].default_value = (0.2, 0.6, 0.2, 1.0)  # Green
                bsdf.inputs["Roughness"].default_value = 0.8

            # Add material to object
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Material could not be applied: {e}")

    def generate_forest(
        self,
        tree_count: int = 50,
        area_size: float = 100.0,
        tree_types: List[str] = None,
        seed: int = 42,
    ) -> List[bpy.types.Object]:
        """Generate a forest"""

        if tree_types is None:
            tree_types = ["oak", "pine", "maple"]

        np.random.seed(seed)
        trees = []

        for i in range(tree_count):
            # Random position
            x = np.random.uniform(-area_size / 2, area_size / 2)
            y = np.random.uniform(-area_size / 2, area_size / 2)
            z = 0  # On terrain surface

            # Random tree type
            tree_type = np.random.choice(tree_types)

            # Random scaling
            scale = np.random.uniform(0.7, 1.3)

            # Generate tree
            tree = self.generate_tree(
                tree_type=tree_type, seed=seed + i, position=(x, y, z), scale=scale
            )

            if tree:
                trees.append(tree)

        self.logger.info(f"✅ Forest generated: {len(trees)} trees")
        return trees

    def get_available_tree_types(self) -> List[str]:
        """Get available tree types"""
        return list(self.tree_types.keys())


class SimpleTreeFactory:
    """Simple Tree Factory - replaces TreeFactory"""

    def __init__(self, seed: int, coarse: bool = False):
        self.seed = seed
        self.coarse = coarse
        self.generator = SimpleTreeGenerator()
        self.logger = logging.getLogger(__name__)

    def create_asset(
        self, params: Dict[str, Any] = None, **kwargs
    ) -> Optional[bpy.types.Object]:
        """Create tree asset"""

        if params is None:
            params = {}

        # Extract parameters
        tree_type = params.get("tree_type", "oak")
        position = params.get("position", (0, 0, 0))
        scale = params.get("scale", 1.0)

        # Generate tree
        tree = self.generator.generate_tree(
            tree_type=tree_type, seed=self.seed, position=position, scale=scale
        )

        return tree

    def finalize_placeholders(self, objs: List[bpy.types.Object]):
        """Finalize placeholders - for compatibility"""
        # Simplified implementation - no special finalization needed
        self.logger.info(f"✅ {len(objs)} tree placeholders finalized")
        pass

    def apply(self, obj: bpy.types.Object, **kwargs) -> bpy.types.Object:
        """Apply tree to object - for compatibility"""
        return obj

    def spawn_placeholder(
        self, i: int, loc: Tuple[float, float, float], rot: Any
    ) -> Optional[bpy.types.Object]:
        """Create placeholder object for placement system - following original Infinigen pattern"""
        try:
            # Use original Infinigen concept: Simple cube as placeholder
            # This is much more efficient for boolean operations
            placeholder = butil.spawn_cube(size=4, location=loc, name=f"Tree_Placeholder_{i}")
            placeholder.rotation_euler = rot

            # Tagging for Infinigen compatibility
            placeholder["is_placeholder"] = True
            placeholder["tree_type"] = "placeholder"
            placeholder["coarse_mesh_placeholder"] = True

            return placeholder

        except Exception as e:
            self.logger.error(f"Error creating placeholder: {e}")
            return None


# Dependencies for Agents
def get_simple_tree_generator() -> SimpleTreeGenerator:
    """Get simple tree generator"""
    return SimpleTreeGenerator()


def get_simple_tree_factory(seed: int, coarse: bool = False) -> SimpleTreeFactory:
    """Get simple tree factory"""
    return SimpleTreeFactory(seed, coarse)