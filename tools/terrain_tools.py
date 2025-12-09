#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain Tools for TerrainEngineerAgent
Clean separation: Engine in infinigen/terrain/, Tools here for Agent
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import bpy
import numpy as np

# Infinigen Terrain Engine import
from infinigen.terrain.engine import ModernTerrainEngine, TerrainConfig, TerrainType

logger = logging.getLogger(__name__)


class TerrainTools:
    """Tools for TerrainEngineerAgent - Uses Infinigen Terrain Engine"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Terrain Engine created on demand
        self.terrain_engine = None

        self.logger.info("✅ Terrain Tools initialized")

    def generate_coarse_terrain(
        self, terrain_type: str = "hills", seed: int = 42, resolution: int = 128
    ) -> Dict[str, Any]:
        """Generate coarse terrain for Coarse-Task"""
        try:
            self.logger.info(
                f"Generating coarse terrain: {terrain_type} (seed: {seed})"
            )

            # Create terrain engine with configuration
            config = TerrainConfig(
                terrain_type=TerrainType(terrain_type) if hasattr(TerrainType, terrain_type.upper()) else TerrainType.HILLS,
                resolution=resolution,
                seed=seed,
                use_pytorch_geometric=False,  # Disable for coarse terrain
                use_kernels=False,  # Disable for coarse terrain
                use_duckdb_storage=False  # Disable for coarse terrain
            )
            
            self.terrain_engine = ModernTerrainEngine(config, self.device)
            result = self.terrain_engine.generate_terrain()

            if result["success"]:
                self.logger.info(
                    f"✅ Coarse terrain generated: {result['metadata']['generation_time']:.2f}s"
            )
            return {
                "success": True,
                    "terrain_mesh": result["terrain_object"],
                    "terrain_id": f"{terrain_type}_{seed}",
                    "height_map": result["height_map"],
                    "normal_map": result["normal_map"],
                    "displacement_map": result["displacement_map"],
                    "roughness_map": result["roughness_map"],
                    "ao_map": result["ao_map"],
                    "generation_time": result["metadata"]["generation_time"],
                    "vertices_count": result["metadata"]["terrain_size"],
                    "faces_count": result["metadata"]["terrain_size"] // 2,
                }
            else:
                self.logger.error(
                    f"❌ Coarse terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                return {"success": False, "error": result.get("error", "Unknown error")}

        except Exception as e:
            self.logger.error(f"Error generating coarse terrain: {e}")
            return {"success": False, "error": str(e)}

    def generate_fine_terrain(
        self, terrain_type: str = "mountain", seed: int = 42, resolution: int = 512
    ) -> Dict[str, Any]:
        """Generate fine terrain for Fine-Task"""
        try:
            self.logger.info(
                f"Generating fine terrain: {terrain_type} (seed: {seed})"
            )

            # Create terrain engine with configuration
            config = TerrainConfig(
                terrain_type=TerrainType(terrain_type) if hasattr(TerrainType, terrain_type.upper()) else TerrainType.MOUNTAIN,
                resolution=resolution,
                seed=seed,
                use_pytorch_geometric=True,  # Enable for fine terrain
                use_kernels=True,  # Enable for fine terrain
                use_duckdb_storage=True  # Enable for fine terrain
            )
            
            self.terrain_engine = ModernTerrainEngine(config, self.device)
            result = self.terrain_engine.generate_terrain()

            if result["success"]:
                self.logger.info(
                    f"✅ Fine terrain generated: {result['metadata']['generation_time']:.2f}s"
                )
            return {
                "success": True,
                    "terrain_mesh": result["terrain_object"],
                    "terrain_id": f"{terrain_type}_{seed}",
                    "height_map": result["height_map"],
                    "normal_map": result["normal_map"],
                    "displacement_map": result["displacement_map"],
                    "roughness_map": result["roughness_map"],
                    "ao_map": result["ao_map"],
                    "generation_time": result["metadata"]["generation_time"],
                    "vertices_count": result["metadata"]["terrain_size"],
                    "faces_count": result["metadata"]["terrain_size"] // 2,
                }
            else:
                self.logger.error(
                    f"❌ Fine terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                return {"success": False, "error": result.get("error", "Unknown error")}

        except Exception as e:
            self.logger.error(f"Error generating fine terrain: {e}")
            return {"success": False, "error": str(e)}

    def optimize_terrain_performance(
        self, terrain_mesh: Any, optimization_level: str = "medium"
    ) -> Dict[str, Any]:
        """Optimize terrain performance"""
        try:
            self.logger.info(f"Optimizing terrain performance: {optimization_level}")

            # Simple optimization based on level
            if optimization_level == "low":
                # Minimal optimization
                return {
                    "success": True,
                    "optimization_level": optimization_level,
                    "mesh_simplification": 0.8,
                    "texture_compression": True,
                    "lod_levels": 2,
                }
            elif optimization_level == "medium":
                # Balanced optimization
                return {
                    "success": True,
                    "optimization_level": optimization_level,
                    "mesh_simplification": 0.9,
                    "texture_compression": True,
                    "lod_levels": 3,
                }
            else:  # high
                # Maximum optimization
                return {
                    "success": True,
                    "optimization_level": optimization_level,
                    "mesh_simplification": 1.0,
                    "texture_compression": False,
                    "lod_levels": 4,
                }

        except Exception as e:
            self.logger.error(f"Error optimizing terrain: {e}")
            return {"success": False, "error": str(e)}

    def get_terrain_recommendations(
        self, scene_type: str, performance_requirements: str = "medium"
    ) -> Dict[str, Any]:
        """Get AI-powered terrain recommendations"""
        try:
            recommendations = {
                "scene_type": scene_type,
                "performance_requirements": performance_requirements,
                "recommended_terrain_type": self._get_terrain_type_recommendation(
                    scene_type
                ),
                "detail_settings": self._get_detail_settings(performance_requirements),
                "optimization_tips": self._get_terrain_optimization_tips(
                    performance_requirements
                ),
                "memory_usage_estimate": self._get_memory_estimate(
                    performance_requirements
                ),
            }

            return {"success": True, "recommendations": recommendations}

        except Exception as e:
            self.logger.error(f"Failed to get terrain recommendations: {e}")
            return {"success": False, "error": str(e)}

    def _get_terrain_type_recommendation(self, scene_type: str) -> str:
        """Get terrain type recommendation based on scene type"""
        terrain_map = {
            "forest": "mountain",
            "desert": "plateau",
            "arctic": "mountain",
            "coastal": "hills",
            "volcanic": "mountain",
            "canyon": "valley",
            "river": "valley",
        }
        return terrain_map.get(scene_type, "hills")

    def _get_detail_settings(self, performance_requirements: str) -> Dict[str, Any]:
        """Get detail settings based on performance requirements"""
        detail_map = {
            "low": {
                "heightmap_resolution": 512,
                "mesh_density": 0.5,
                "texture_resolution": 1024,
                "erosion_iterations": 10,
            },
            "medium": {
                "heightmap_resolution": 1024,
                "mesh_density": 1.0,
                "texture_resolution": 2048,
                "erosion_iterations": 20,
            },
            "high": {
                "heightmap_resolution": 2048,
                "mesh_density": 1.5,
                "texture_resolution": 4096,
                "erosion_iterations": 50,
            },
        }
        return detail_map.get(performance_requirements, detail_map["medium"])

    def _get_terrain_optimization_tips(
        self, performance_requirements: str
    ) -> List[str]:
        """Get terrain optimization tips"""
        tips = {
            "low": [
                "Use lower resolution heightmaps",
                "Simplify mesh geometry",
                "Use texture compression",
                "Enable LOD (Level of Detail)",
            ],
            "medium": [
                "Balance detail vs performance",
                "Use instancing for repeated elements",
                "Optimize texture atlasing",
                "Enable frustum culling",
            ],
            "high": [
                "Use high-resolution heightmaps",
                "Enable GPU acceleration",
                "Use advanced shaders",
                "Implement dynamic LOD",
            ],
        }
        return tips.get(performance_requirements, tips["medium"])

    def _get_memory_estimate(self, performance_requirements: str) -> Dict[str, str]:
        """Get memory usage estimate"""
        memory_map = {
            "low": {"estimated_ram": "2-4 GB", "estimated_vram": "1-2 GB"},
            "medium": {"estimated_ram": "4-8 GB", "estimated_vram": "2-4 GB"},
            "high": {"estimated_ram": "8-16 GB", "estimated_vram": "4-8 GB"},
        }
        return memory_map.get(performance_requirements, memory_map["medium"])

    def cleanup(self):
        """Cleanup resources"""
        if self.terrain_engine:
            self.terrain_engine.cleanup()
        self.logger.info("Terrain tools cleanup completed")