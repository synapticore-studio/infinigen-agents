# Addon Manager Agent - For managing Blender addons and extensions
import logging
from typing import Dict, List

from pydantic_ai import Agent, RunContext

from config.model_factory import get_model
from deps.blender_deps import BlenderConnectionDep
from tools.blender_tools import BlenderOpsDep

logger = logging.getLogger(__name__)

# Define dependencies type for the agent
class AddonManagerDeps:
    """Dependencies for the Addon Manager Agent"""
    def __init__(
        self,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
    ):
        self.blender_ops = blender_ops
        self.blender_conn = blender_conn


# Create the Agent directly using pydantic-ai patterns
addon_manager_agent = Agent(
    get_model(),
    result_type=Dict,
    deps_type=AddonManagerDeps,
    system_prompt="""You are an expert Blender addon manager. Your responsibilities include:

1. **Addon Installation & Management**: Install, enable, disable, and configure Blender addons
2. **Dependency Resolution**: Ensure all addon dependencies are properly installed
3. **Addon Validation**: Verify addons are working correctly
4. **Troubleshooting**: Diagnose and fix addon-related issues
5. **Configuration**: Set up addons with optimal settings for Infinigen

**Key Addons for Infinigen:**
- **Real Snow**: Snow effects and particle systems
- **FLIP Fluids**: Fluid simulation capabilities  
- **Ant Landscape**: Terrain generation tools
- **Differential Growth**: Organic shape generation

**Best Practices:**
- Always check addon status before operations
- Validate dependencies before enabling addons
- Use appropriate fail modes (warn/fatal) based on addon importance
- Provide clear error messages and solutions
- Test addon functionality after installation

**Error Handling:**
- Required addons: Use 'fatal' mode - generation fails if not available
- Optional addons: Use 'warn' mode - log warnings but continue
- Always provide fallback options when possible

You have access to comprehensive addon management tools and can handle both online and offline installation methods.""",
)


@addon_manager_agent.tool
async def install_addon(
    ctx: RunContext[AddonManagerDeps],
    addon_name: str,
) -> Dict:
    """Install a Blender addon
    
    Args:
        ctx: The run context with dependencies
        addon_name: Name of the addon to install
        
    Returns:
        Dictionary with installation status and details
    """
    try:
        if not ctx.deps.blender_conn.is_connected:
            return {
                "success": False,
                "message": "Blender not connected",
                "error": "No Blender connection available",
            }

        # Use BlenderOps to install addon
        result = ctx.deps.blender_ops.install_addon(addon_name)

        return {
            "success": result,
            "message": f"Addon {addon_name} installation {'succeeded' if result else 'failed'}",
            "addon_name": addon_name,
        }

    except Exception as e:
        logger.error(f"Error installing addon {addon_name}: {e}")
        return {
            "success": False,
            "message": f"Error installing addon {addon_name}",
            "error": str(e),
        }


@addon_manager_agent.tool
async def setup_required_addons(
    ctx: RunContext[AddonManagerDeps],
) -> Dict[str, Dict]:
    """Setup all required addons for Infinigen
    
    Args:
        ctx: The run context with dependencies
        
    Returns:
        Dictionary mapping addon names to their installation results
    """
    required_addons = ["real_snow", "flip_fluids", "ant_landscape"]
    results = {}

    for addon_name in required_addons:
        try:
            result = await install_addon(ctx, addon_name)
            results[addon_name] = result

            if not result["success"]:
                logger.warning(f"Failed to setup required addon: {addon_name}")

        except Exception as e:
            logger.error(f"Error setting up addon {addon_name}: {e}")
            results[addon_name] = {
                "success": False,
                "message": f"Error setting up addon {addon_name}",
                "error": str(e),
            }

    return results
