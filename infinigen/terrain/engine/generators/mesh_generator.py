#!/usr/bin/env python3
"""
Terrain Mesh Generator
Creates 3D meshes from height maps using modern Blender integration
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

# Blender import with fallback
try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None

# Trimesh import with fallback
try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

from .base_generator import BaseTerrainGenerator, GeneratorConfig

logger = logging.getLogger(__name__)


class TerrainMeshGenerator(BaseTerrainGenerator):
    """Generates 3D terrain meshes from height maps"""

    def __init__(self, config: GeneratorConfig, device: str = "cpu"):
        super().__init__(config, device)
        self._set_random_seed()

    def create_terrain_mesh(self, height_map: np.ndarray) -> Optional[Any]:
        """Create terrain mesh from height map using modern Blender integration"""
        try:
            self.logger.info("Creating terrain mesh from height map")

            # Resize heightmap to target resolution for camera placement compatibility
            target_resolution = 128  # Standard resolution for camera placement
            if height_map.shape != (target_resolution, target_resolution):
                height_map = self._resize_height_map(height_map, target_resolution)

            # Create mesh using modern Blender integration
            terrain_obj = self._create_blender_mesh(height_map)

            if terrain_obj:
                self.logger.info(f"✅ Terrain mesh created: {terrain_obj.name}")
                return terrain_obj
            else:
                self.logger.warning("⚠️ Failed to create terrain mesh, using fallback")
                return self._create_fallback_mesh(height_map)

        except Exception as e:
            self.logger.error(f"Error creating terrain mesh: {e}")
            return self._create_fallback_mesh(height_map)

    def _resize_height_map(
        self, height_map: np.ndarray, target_resolution: int
    ) -> np.ndarray:
        """Resize height map to target resolution"""
        try:
            import cv2

            return cv2.resize(height_map, (target_resolution, target_resolution))
        except ImportError:
            # Fallback without cv2
            from scipy.ndimage import zoom

            scale_factor = target_resolution / height_map.shape[0]
            return zoom(height_map, scale_factor)

    def _create_blender_mesh(self, height_map: np.ndarray) -> Optional[Any]:
        """Create Blender mesh from height map using modern methods"""
        try:
            height, width = height_map.shape

            # Create vertices from heightmap
            vertices = []
            for y in range(height):
                for x in range(width):
                    z = height_map[y, x] * 10  # Scale height
                    vertices.append([x - width / 2, y - height / 2, z])

            # Create faces (triangles)
            faces = []
            for y in range(height - 1):
                for x in range(width - 1):
                    # Two triangles per quad
                    v1 = y * width + x
                    v2 = y * width + (x + 1)
                    v3 = (y + 1) * width + x
                    v4 = (y + 1) * width + (x + 1)

                    # First triangle
                    faces.append([v1, v2, v3])
                    # Second triangle
                    faces.append([v2, v4, v3])

            # Create new mesh in Blender
            mesh_name = f"{self.config.terrain_type}_terrain_{self.config.seed}"
            mesh = bpy.data.meshes.new(mesh_name)
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            # Create object
            terrain_obj = bpy.data.objects.new(mesh_name, mesh)
            bpy.context.collection.objects.link(terrain_obj)

            # Add modern material
            self._add_modern_material(terrain_obj)

            return terrain_obj

        except Exception as e:
            self.logger.error(f"Error creating Blender mesh: {e}")
            return None

    def _add_modern_material(self, terrain_obj: Any):
        """Add modern material to terrain object"""
        try:
            material_name = f"TerrainMaterial_{self.config.terrain_type}"
            material = bpy.data.materials.new(name=material_name)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Clear default nodes
            nodes.clear()

            # Add Principled BSDF
            principled = nodes.new(type="ShaderNodeBsdfPrincipled")

            # Set material properties based on terrain type
            if self.config.terrain_type == "mountain":
                principled.inputs["Base Color"].default_value = (
                    0.4,
                    0.4,
                    0.3,
                    1.0,
                )  # Rock color
                principled.inputs["Roughness"].default_value = 0.9
            elif self.config.terrain_type == "hills":
                principled.inputs["Base Color"].default_value = (
                    0.4,
                    0.6,
                    0.3,
                    1.0,
                )  # Green
                principled.inputs["Roughness"].default_value = 0.8
            elif self.config.terrain_type == "valley":
                principled.inputs["Base Color"].default_value = (
                    0.3,
                    0.5,
                    0.2,
                    1.0,
                )  # Dark green
                principled.inputs["Roughness"].default_value = 0.7
            elif self.config.terrain_type == "desert":
                principled.inputs["Base Color"].default_value = (
                    0.8,
                    0.7,
                    0.4,
                    1.0,
                )  # Sand color
                principled.inputs["Roughness"].default_value = 0.9
            else:
                principled.inputs["Base Color"].default_value = (
                    0.4,
                    0.6,
                    0.3,
                    1.0,
                )  # Default green
                principled.inputs["Roughness"].default_value = 0.8

            principled.inputs["Metallic"].default_value = 0.0

            # Add Material Output
            output = nodes.new(type="ShaderNodeOutputMaterial")

            # Connect nodes
            links.new(principled.outputs["BSDF"], output.inputs["Surface"])

            # Assign material
            terrain_obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not add modern material: {e}")

    def _create_fallback_mesh(self, height_map: np.ndarray) -> Optional[Any]:
        """Create fallback mesh if main method fails"""
        try:
            # Create simple cube as fallback
            bpy.ops.mesh.primitive_cube_add()
            terrain_obj = bpy.context.active_object
            terrain_obj.name = (
                f"{self.config.terrain_type}_terrain_{self.config.seed}_fallback"
            )

            # Scale based on height map
            scale_z = np.max(height_map) - np.min(height_map)
            terrain_obj.scale = (2, 2, scale_z / 10)

            return terrain_obj

        except Exception as e:
            self.logger.error(f"Error creating fallback mesh: {e}")
            return None

    def generate(self, **kwargs) -> Any:
        """Generate terrain data - implemented by subclasses"""
        height_map = kwargs.get("height_map")
        if height_map is None:
            # Generate a default height map
            from .map_generator import GeneratorConfig, TerrainMapGenerator

            map_config = GeneratorConfig(
                terrain_type=self.config.terrain_type,
                resolution=self.config.resolution,
                seed=self.config.seed,
            )
            map_generator = TerrainMapGenerator(map_config, self.device)
            height_map = map_generator.generate_height_map()

        return self.create_terrain_mesh(height_map)
