# Addon Manager Agent - For managing Blender addons and extensions
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent

from deps.blender_deps import BlenderConnectionDep
from tools.blender_tools import BlenderOpsDep

logger = logging.getLogger(__name__)


class AddonManagerAgent(BaseModel):
    """AI Agent for managing Blender addons and extensions"""

    def __init__(self, **data):
        super().__init__(**data)

        # Create Pydantic-AI agent
        self.agent = Agent(
            "openai:gpt-4o-mini",
            result_type=Dict,
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

    async def install_addon(
        self,
        addon_name: str,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
    ) -> Dict:
        """Install a Blender addon"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "message": "Blender not connected",
                    "error": "No Blender connection available",
                }

            # Use BlenderOps to install addon
            result = blender_ops.install_addon(addon_name)

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

    async def setup_required_addons(
        self, blender_ops: BlenderOpsDep, blender_conn: BlenderConnectionDep
    ) -> Dict[str, Dict]:
        """Setup all required addons for Infinigen"""
        required_addons = ["real_snow", "flip_fluids", "ant_landscape"]
        results = {}

        for addon_name in required_addons:
            try:
                result = await self.install_addon(addon_name, blender_ops, blender_conn)
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
