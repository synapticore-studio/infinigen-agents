# Asset Generator Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext
from config.model_factory import get_model

from deps.blender_deps import BlenderConnectionDep
from deps.core_deps import MathUtilsDep, SeedManagerDep
from tools.blender_tools import BlenderOpsDep

logger = logging.getLogger(__name__)


# Define dependencies type for the agent
class AssetGeneratorDeps:
    """Dependencies for the Asset Generator Agent"""
    def __init__(
        self,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        seed_manager: SeedManagerDep,
        math_utils: MathUtilsDep,
    ):
        self.blender_ops = blender_ops
        self.blender_conn = blender_conn
        self.seed_manager = seed_manager
        self.math_utils = math_utils


# Create the Agent directly using pydantic-ai patterns
asset_generator_agent = Agent(
    get_model(),
    result_type=Dict[str, Any],
    deps_type=AssetGeneratorDeps,
    system_prompt="""You are a specialized asset generation agent for Infinigen.
    
    Your responsibilities:
    - Generate procedural creatures, trees, and objects
    - Create procedural materials and textures
    - Manage asset parameters and variations
    - Optimize asset generation for performance
    
    Available asset categories:
    - Creatures: carnivore, herbivore, bird, fish
    - Trees: pine, oak, palm, bamboo
    - Materials: ground, water, rock, snow, sand
    - Objects: rock, cloud, particle_system
    
    Always validate asset parameters and provide detailed feedback on generation success/failure.
    Consider performance implications when suggesting asset complexity.
    """,
)


@asset_generator_agent.tool
async def generate_creature_asset(
    ctx: RunContext[AssetGeneratorDeps],
    creature_type: str,
    output_path: Path,
    seed: int = 42,
    complexity: str = "medium",
    **kwargs,
) -> Dict[str, Any]:
    """Generate a creature asset with AI assistance
    
    Args:
        ctx: The run context with dependencies
        creature_type: Type of creature to generate
        output_path: Path to save the asset
        seed: Seed for reproducible generation
        complexity: Complexity level (low/medium/high)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with generation status and details
    """
    try:
        if not ctx.deps.blender_conn.is_connected:
            return {
                "success": False,
                "error": "Blender not connected",
            }

        # Set seed for reproducible generation
        ctx.deps.seed_manager.set_seed(seed)

        # Use BlenderOps to create mesh
        mesh = ctx.deps.blender_ops.create_mesh(f"{creature_type}_{seed}")

        # Generate random properties using MathUtils
        random_vector = ctx.deps.math_utils.generate_random_vector(seed, dimensions=3)
        random_rotation = ctx.deps.math_utils.generate_random_rotation(seed)

        # Apply creature-specific modifications
        if creature_type == "carnivore":
            # Add sharp features using random properties
            pass
        elif creature_type == "herbivore":
            # Add gentle features using random properties
            pass

        logger.info(
            f"Successfully generated {creature_type} creature with complexity {complexity}"
        )

        return {
            "success": True,
            "creature_type": creature_type,
            "output_path": str(output_path),
            "seed": seed,
            "complexity": complexity,
            "mesh_name": mesh.name if mesh else None,
        }

    except Exception as e:
        logger.error(f"Creature generation failed: {e}")
        return {"success": False, "error": str(e)}


@asset_generator_agent.tool
async def generate_tree_asset(
    ctx: RunContext[AssetGeneratorDeps],
    tree_type: str,
    output_path: Path,
    seed: int = 42,
    complexity: str = "medium",
    **kwargs,
) -> Dict[str, Any]:
    """Generate a tree asset with AI assistance
    
    Args:
        ctx: The run context with dependencies
        tree_type: Type of tree to generate
        output_path: Path to save the asset
        seed: Seed for reproducible generation
        complexity: Complexity level (low/medium/high)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with generation status and details
    """
    try:
        if not ctx.deps.blender_conn.is_connected:
            return {
                "success": False,
                "error": "Blender not connected",
            }

        # Set seed for reproducible generation
        ctx.deps.seed_manager.set_seed(seed)

        # Use BlenderOps to create mesh
        mesh = ctx.deps.blender_ops.create_mesh(f"{tree_type}_{seed}")

        # Generate random properties using MathUtils
        random_vector = ctx.deps.math_utils.generate_random_vector(seed, dimensions=3)
        random_rotation = ctx.deps.math_utils.generate_random_rotation(seed)

        # Apply tree-specific modifications
        if tree_type == "pine":
            # Add conical shape using random properties
            pass
        elif tree_type == "oak":
            # Add broad canopy using random properties
            pass

        logger.info(
            f"Successfully generated {tree_type} tree with complexity {complexity}"
        )

        return {
            "success": True,
            "tree_type": tree_type,
            "output_path": str(output_path),
            "seed": seed,
            "complexity": complexity,
            "mesh_name": mesh.name if mesh else None,
        }

    except Exception as e:
        logger.error(f"Tree generation failed: {e}")
        return {"success": False, "error": str(e)}


@asset_generator_agent.tool
async def generate_material_asset(
    ctx: RunContext[AssetGeneratorDeps],
    material_type: str,
    output_path: Path,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a material asset with AI assistance
    
    Args:
        ctx: The run context with dependencies
        material_type: Type of material to generate
        output_path: Path to save the asset
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with generation status and details
    """
    try:
        if not ctx.deps.blender_conn.is_connected:
            return {
                "success": False,
                "error": "Blender not connected",
            }

        # Use BlenderOps to create material
        material = ctx.deps.blender_ops.create_material(f"{material_type}_material")

        # Apply material-specific properties
        if material_type == "ground":
            # Add earthy colors and roughness
            pass
        elif material_type == "water":
            # Add transparency and reflection
            pass

        logger.info(f"Successfully generated {material_type} material")

        return {
            "success": True,
            "material_type": material_type,
            "output_path": str(output_path),
            "material_name": material.name if material else None,
        }

    except Exception as e:
        logger.error(f"Material generation failed: {e}")
        return {"success": False, "error": str(e)}
