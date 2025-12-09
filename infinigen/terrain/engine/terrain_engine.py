#!/usr/bin/env python3
"""
Modern Terrain Engine - Main orchestrator
Coordinates all terrain generation components using modern tech stack
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch

# Blender import with fallback
try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None

from .generators import TerrainMapGenerator, TerrainMeshGenerator
from .processors import KernelsProcessor, PyTorchGeometricProcessor
from .storage import DuckDBSpatialManager

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Terrain types supported by the modern engine"""

    MOUNTAIN = "mountain"
    HILLS = "hills"
    VALLEY = "valley"
    PLATEAU = "plateau"
    DESERT = "desert"
    OCEAN = "ocean"


@dataclass
class TerrainConfig:
    """Configuration for terrain generation"""

    terrain_type: TerrainType = TerrainType.MOUNTAIN
    resolution: int = 256
    seed: int = 42
    bounds: tuple = (-50, 50, -50, 50, 0, 100)
    use_pytorch_geometric: bool = True
    use_kernels: bool = True
    use_duckdb_storage: bool = True
    enable_geometry_baking: bool = True


class ModernTerrainEngine:
    """Complete modern terrain generation engine with full tech stack integration"""

    def __init__(self, config: TerrainConfig = None, device: str = "cpu"):
        self.config = config or TerrainConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize components with modern tech stack
        self.map_generator = TerrainMapGenerator(self.config, self.device)
        self.mesh_generator = TerrainMeshGenerator(self.config, self.device)
        self.pytorch_processor = PyTorchGeometricProcessor(self.config, self.device)
        self.kernels_processor = KernelsProcessor(self.config, self.device)
        self.spatial_manager = (
            DuckDBSpatialManager() if self.config.use_duckdb_storage else None
        )
            # Using Infinigen's native systems instead of custom Blender4Integration

        # Initialize random seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def generate_terrain(self, **kwargs) -> Dict[str, Any]:
        """Generate complete terrain using modern tech stack"""
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating MODERN terrain: {self.config.terrain_type.value}"
            )

            # 1. Generate height map using TerrainMapGenerator
            height_map = self.map_generator.generate_height_map()
            if height_map is None:
                return {"success": False, "error": "Failed to generate height map"}

            # 2. Process with PyTorch Geometric if enabled
            if self.config.use_pytorch_geometric:
                height_map = self.pytorch_processor.process_height_map(height_map)

            # 3. Process with Kernels if enabled
            if self.config.use_kernels:
                height_map = self.kernels_processor.process_height_map(height_map)

            # 4. Generate additional maps
            normal_map = self.map_generator.generate_normal_map(height_map)
            displacement_map = self.map_generator.generate_displacement_map(height_map)
            roughness_map = self.map_generator.generate_roughness_map(height_map)
            ao_map = self.map_generator.generate_ao_map(height_map)

            # 5. Create terrain mesh using TerrainMeshGenerator
            terrain_obj = self.mesh_generator.create_terrain_mesh(height_map)
            if not terrain_obj:
                return {"success": False, "error": "Failed to create terrain mesh"}

            # 6. Apply Infinigen's native terrain features
            self._apply_infinigen_features(terrain_obj)

            # 9. Store in DuckDB if enabled
            if self.spatial_manager:
                self.spatial_manager.store_terrain_data(terrain_obj, height_map)

            generation_time = time.time() - start_time

            return {
                "success": True,
                "terrain_object": terrain_obj,
                "height_map": height_map,
                "normal_map": normal_map,
                "displacement_map": displacement_map,
                "roughness_map": roughness_map,
                "ao_map": ao_map,
                "metadata": {
                    "terrain_type": self.config.terrain_type.value,
                    "seed": self.config.seed,
                    "resolution": self.config.resolution,
                    "generation_time": generation_time,
                    "terrain_size": (
                        len(terrain_obj.data.vertices)
                        if terrain_obj and terrain_obj.data
                        else 0
                    ),
                    "tech_stack": {
                        "pytorch_geometric": self.config.use_pytorch_geometric,
                        "kernels": self.config.use_kernels,
                        "duckdb": self.config.use_duckdb_storage,
                        "blender_4_5_3": True,
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating terrain: {e}")
            return {"success": False, "error": str(e)}

    def _apply_infinigen_features(self, terrain_obj: Any):
        """Apply Infinigen's native terrain features"""
        try:
            if not BLENDER_AVAILABLE or not terrain_obj:
                return

            # Use Infinigen's native material system
            from infinigen.core.surface import assign_material
            from infinigen.core.tagging import tag_object
            from infinigen.core.util.organization import Tags, Materials

            # Tag the terrain object
            tag_object(terrain_obj, Tags.Terrain)
            
            # Create or get material for terrain
            material_name = f"TerrainMaterial_{self.config.terrain_type.value}"
            if material_name in bpy.data.materials:
                material = bpy.data.materials[material_name]
            else:
                # Create new material
                material = bpy.data.materials.new(name=material_name)
                material.use_nodes = True
                
                # Set up basic material properties
                nodes = material.node_tree.nodes
                principled = nodes.get("Principled BSDF")
                if principled:
                    # Set terrain-appropriate colors
                    if self.config.terrain_type.value == "mountain":
                        principled.inputs["Base Color"].default_value = (0.4, 0.4, 0.3, 1.0)  # Rock
                        principled.inputs["Roughness"].default_value = 0.9
                    elif self.config.terrain_type.value == "hills":
                        principled.inputs["Base Color"].default_value = (0.4, 0.6, 0.3, 1.0)  # Green
                        principled.inputs["Roughness"].default_value = 0.8
                    else:
                        principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.4, 1.0)  # Default
                        principled.inputs["Roughness"].default_value = 0.8
            
            # Apply the material to terrain object
            assign_material(terrain_obj, material)
            
            self.logger.info("✅ Applied Infinigen's native terrain features")

        except Exception as e:
            self.logger.warning(f"Could not apply Infinigen features: {e}")

    def cleanup(self):
        """Cleanup resources"""
        if self.spatial_manager:
            self.spatial_manager.close()
        self.logger.info("Terrain engine cleanup completed")
