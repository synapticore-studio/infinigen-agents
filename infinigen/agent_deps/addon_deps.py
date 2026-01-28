# Addon Dependencies - Pydantic-AI compatible
from dataclasses import dataclass
from typing import Dict, List, Optional

import bpy

# Simple dependency injection


@dataclass
class AddonManager:
    """Core addon management functionality"""

    def check_addon_status(self, addon_name: str) -> Dict[str, bool]:
        """Check if addon is installed and enabled"""
        try:
            # Check if addon is in preferences
            installed = addon_name in bpy.context.preferences.addons
            enabled = installed and bpy.context.preferences.addons[addon_name].use

            return {"installed": installed, "enabled": enabled, "module": addon_name}
        except Exception as e:
            return {"installed": False, "enabled": False, "error": str(e)}

    def enable_addon(self, addon_name: str) -> bool:
        """Enable a Blender addon"""
        try:
            if addon_name not in bpy.context.preferences.addons:
                return False

            bpy.ops.preferences.addon_enable(module=addon_name)
            return True
        except Exception:
            return False

    def disable_addon(self, addon_name: str) -> bool:
        """Disable a Blender addon"""
        try:
            if addon_name not in bpy.context.preferences.addons:
                return True

            bpy.ops.preferences.addon_disable(module=addon_name)
            return True
        except Exception:
            return False


@dataclass
class RequiredAddons:
    """Required addons for Infinigen"""

    addons: Dict[str, str] = None

    def __post_init__(self):
        if self.addons is None:
            self.addons = {
                "real_snow": "object.real_snow",
                "flip_fluids": "flip_fluids_addon",
                "antlandscape": "antlandscape",
                "mtree": "mtree",
                "treegen": "treegen",
            }

    def get_module_name(self, addon_name: str) -> Optional[str]:
        """Get the actual module name for an addon"""
        return self.addons.get(addon_name)

    def list_addons(self) -> List[str]:
        """List all required addon names"""
        return list(self.addons.keys())


@dataclass
class OptionalAddons:
    """Optional addons for Infinigen"""

    addons: Dict[str, str] = None

    def __post_init__(self):
        if self.addons is None:
            self.addons = {"differential_growth": "differential_growth"}

    def get_module_name(self, addon_name: str) -> Optional[str]:
        """Get the actual module name for an addon"""
        return self.addons.get(addon_name)

    def list_addons(self) -> List[str]:
        """List all optional addon names"""
        return list(self.addons.keys())


# Pydantic-AI Dependency Functions
def get_addon_manager() -> AddonManager:
    """Get addon manager instance"""
    return AddonManager()


def get_required_addons() -> RequiredAddons:
    """Get required addons configuration"""
    return RequiredAddons()


def get_optional_addons() -> OptionalAddons:
    """Get optional addons configuration"""
    return OptionalAddons()


# Pydantic-AI Depends objects for agents
AddonManagerDep = get_addon_manager
RequiredAddonsDep = get_required_addons
OptionalAddonsDep = get_optional_addons
