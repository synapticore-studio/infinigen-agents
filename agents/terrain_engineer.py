# Terrain Engineer Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic_ai import Agent, RunContext

from deps.core_deps import SeedManagerDep, ValidationManagerDep
from config.model_factory import get_model
from tools.file_tools import FileManagerDep, LoggerDep

logger = logging.getLogger(__name__)


# Define dependencies type for the agent
class TerrainEngineerDeps:
    """Dependencies for the Terrain Engineer Agent"""
    def __init__(
        self,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
        seed_manager: SeedManagerDep,
        validation_manager: ValidationManagerDep,
    ):
        self.file_manager = file_manager
        self.logger_tool = logger_tool
        self.seed_manager = seed_manager
        self.validation_manager = validation_manager


# Create the Agent directly using pydantic-ai patterns
terrain_engineer_agent = Agent(
    get_model(),
    result_type=Dict[str, Any],
    deps_type=TerrainEngineerDeps,
    system_prompt="""You are a specialized terrain generation agent for Infinigen.
    
    Your responsibilities:
    - Generate realistic terrain with different types
    - Optimize terrain for performance
    - Apply erosion and natural features
    - Manage terrain detail levels
    
    Available terrain types:
    - mountain, canyon, plain, cliff, coast, arctic, volcano, river
    
    Detail levels:
    - coarse: Low resolution, fast generation
    - medium: Balanced quality and performance
    - fine: High resolution, detailed terrain
    
    Always validate terrain parameters and provide detailed feedback on generation success/failure.
    Consider memory and performance implications when suggesting detail levels.
    """,
)


@terrain_engineer_agent.tool
async def generate_terrain(
    ctx: RunContext[TerrainEngineerDeps],
    output_folder: Path,
    scene_seed: int,
    terrain_type: str = "mountain",
    detail_level: str = "medium",
) -> Dict[str, Any]:
    """Generate terrain with AI assistance
    
    Args:
        ctx: The run context with dependencies
        output_folder: Path to the output folder
        scene_seed: Seed for reproducible generation
        terrain_type: Type of terrain to generate
        detail_level: Level of detail (coarse/medium/fine)
        
    Returns:
        Dictionary with terrain generation status and details
    """
    try:
        # Set seed for reproducible generation
        ctx.deps.seed_manager.set_seed(scene_seed)

        # Validate output folder
        folder_validation = ctx.deps.validation_manager.validate_path(
            output_folder, must_exist=False
        )
        if not folder_validation["success"]:
            return folder_validation

        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Generate actual terrain using modern terrain engine
        from tools.terrain_tools import TerrainTools

        terrain_tools = TerrainTools(device="cpu")

        # Generate terrain based on detail level
        if detail_level == "coarse":
            terrain_result = terrain_tools.generate_coarse_terrain(
                terrain_type=terrain_type, seed=scene_seed, resolution=128
            )
        else:  # fine or medium
            terrain_result = terrain_tools.generate_fine_terrain(
                terrain_type=terrain_type, seed=scene_seed, resolution=512
            )

        if not terrain_result["success"]:
            return {
                "success": False,
                "error": terrain_result.get("error", "Terrain generation failed"),
            }

        # Create terrain metadata with actual results
        terrain_metadata = {
            "terrain_type": terrain_type,
            "scene_seed": scene_seed,
            "detail_level": detail_level,
            "output_folder": str(output_folder),
            "status": "generated",
            "terrain_size": terrain_result.get("vertices_count", 0),
            "faces_count": terrain_result.get("faces_count", 0),
            "generation_time": terrain_result.get("generation_time", 0),
            "height_map_shape": (
                terrain_result.get("height_map", np.array([])).shape
                if "height_map" in terrain_result
                else None
            ),
        }

        # Save terrain metadata
        terrain_file = output_folder / f"terrain_{scene_seed}_metadata.json"
        success = ctx.deps.file_manager.save_json(terrain_metadata, terrain_file)

        if success:
            ctx.deps.logger_tool.info(
                f"Successfully generated {terrain_type} terrain with detail level {detail_level}"
            )
            ctx.deps.logger_tool.info(
                f"Terrain vertices: {terrain_result.get('vertices_count', 0)}"
            )
            ctx.deps.logger_tool.info(
                f"Generation time: {terrain_result.get('generation_time', 0):.2f}s"
            )
            return {
                "success": True,
                "terrain_type": terrain_type,
                "detail_level": detail_level,
                "seed": scene_seed,
                "output_folder": str(output_folder),
                "terrain_file": str(terrain_file),
                "terrain_mesh": terrain_result.get("terrain_mesh"),
                "height_map": terrain_result.get("height_map"),
                "vertices_count": terrain_result.get("vertices_count", 0),
                "generation_time": terrain_result.get("generation_time", 0),
            }
        else:
            return {"success": False, "error": "Failed to save terrain metadata"}

    except Exception as e:
        logger.error(f"Terrain generation failed: {e}")
        return {"success": False, "error": str(e)}


@terrain_engineer_agent.tool
async def optimize_terrain(
    ctx: RunContext[TerrainEngineerDeps],
    terrain_folder: Path,
    optimization_level: str = "medium",
) -> Dict[str, Any]:
    """Optimize terrain for performance
    
    Args:
        ctx: The run context with dependencies
        terrain_folder: Path to the terrain folder
        optimization_level: Level of optimization (low/medium/high)
        
    Returns:
        Dictionary with optimization status and details
    """
    try:
        # Get terrain info from metadata
        terrain_file = terrain_folder / "terrain_metadata.json"
        terrain_info = ctx.deps.file_manager.load_json(terrain_file)

        if not terrain_info:
            return {"success": False, "error": "No terrain metadata found"}

        # Apply optimizations based on level
        optimizations = _get_optimization_params(optimization_level)

        # Update terrain metadata with optimizations
        terrain_info["optimizations"] = optimizations
        terrain_info["optimization_level"] = optimization_level

        # Save updated metadata
        success = ctx.deps.file_manager.save_json(terrain_info, terrain_file)

        if success:
            ctx.deps.logger_tool.info(f"Applied terrain optimizations: {optimization_level}")
            return {
                "success": True,
                "terrain_folder": str(terrain_folder),
                "optimization_level": optimization_level,
                "optimizations_applied": optimizations,
            }
        else:
            return {
                "success": False,
                "error": "Failed to save optimization metadata",
            }

    except Exception as e:
        logger.error(f"Terrain optimization failed: {e}")
        return {"success": False, "error": str(e)}


# Helper functions (not agent tools)
def get_terrain_recommendations(
    scene_type: str, performance_requirements: str = "medium"
) -> Dict[str, Any]:
    """Get AI-powered terrain recommendations"""
    try:
        recommendations = {
            "scene_type": scene_type,
            "performance_requirements": performance_requirements,
            "recommended_terrain_type": _get_terrain_type_recommendation(
                scene_type
            ),
            "detail_settings": _get_detail_settings(performance_requirements),
            "optimization_tips": _get_terrain_optimization_tips(
                performance_requirements
            ),
            "memory_usage_estimate": _get_memory_estimate(
                performance_requirements
            ),
        }

        return {"success": True, "recommendations": recommendations}

    except Exception as e:
        logger.error(f"Failed to get terrain recommendations: {e}")
        return {"success": False, "error": str(e)}


def _get_optimization_params(optimization_level: str) -> Dict[str, Any]:
    """Get optimization parameters based on level"""
    optimization_map = {
        "low": {
            "mesh_simplification": 0.8,
            "texture_compression": True,
            "lod_levels": 2,
            "culling_distance": 100,
        },
        "medium": {
            "mesh_simplification": 0.9,
            "texture_compression": True,
            "lod_levels": 3,
            "culling_distance": 200,
        },
        "high": {
            "mesh_simplification": 1.0,
            "texture_compression": False,
            "lod_levels": 4,
            "culling_distance": 500,
        },
    }
    return optimization_map.get(optimization_level, optimization_map["medium"])


def _get_terrain_type_recommendation(scene_type: str) -> str:
    """Get terrain type recommendation based on scene type"""
    terrain_map = {
        "forest": "mountain",
        "desert": "plain",
        "arctic": "mountain",
        "coastal": "coast",
        "volcanic": "volcano",
        "canyon": "canyon",
        "river": "river",
    }
    return terrain_map.get(scene_type, "mountain")


def _get_detail_settings(performance_requirements: str) -> Dict[str, Any]:
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
    performance_requirements: str
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


def _get_memory_estimate(performance_requirements: str) -> Dict[str, str]:
    """Get memory usage estimate"""
    memory_map = {
        "low": {"estimated_ram": "2-4 GB", "estimated_vram": "1-2 GB"},
        "medium": {"estimated_ram": "4-8 GB", "estimated_vram": "2-4 GB"},
        "high": {"estimated_ram": "8-16 GB", "estimated_vram": "4-8 GB"},
    }
    return memory_map.get(performance_requirements, memory_map["medium"])
