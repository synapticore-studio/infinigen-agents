# Blender Dependencies - Pure data only
from dataclasses import dataclass
from typing import Optional

# Simple dependency injection


@dataclass
class BlenderConnection:
    """Blender connection state - pure data only"""

    is_connected: bool = False
    version: str = "4.5.3"
    current_scene: Optional[str] = None
    active_object: Optional[str] = None

    # New Blender 4.5+ features
    eevee_next_available: bool = True
    vulkan_backend_available: bool = False
    geometry_nodes_enhanced: bool = True
    animation_layers_available: bool = True
    cuda_13_support: bool = True
    optix_9_support: bool = True

    def __post_init__(self):
        """Initialize connection state"""
        try:
            # Check if Blender is available
            self.is_connected = hasattr(bpy, "context") and bpy.context is not None
            if self.is_connected:
                self.current_scene = (
                    bpy.context.scene.name if bpy.context.scene else None
                )
                self.active_object = (
                    bpy.context.active_object.name
                    if bpy.context.active_object
                    else None
                )
        except Exception:
            self.is_connected = False

    def get_connection_info(self) -> dict:
        """Get current connection information"""
        return {
            "is_connected": self.is_connected,
            "version": self.version,
            "current_scene": self.current_scene,
            "active_object": self.active_object,
            "eevee_next_available": self.eevee_next_available,
            "vulkan_backend_available": self.vulkan_backend_available,
            "geometry_nodes_enhanced": self.geometry_nodes_enhanced,
            "animation_layers_available": self.animation_layers_available,
            "cuda_13_support": self.cuda_13_support,
            "optix_9_support": self.optix_9_support,
        }


def get_blender_connection() -> BlenderConnection:
    """Get Blender connection instance"""
    return BlenderConnection()


# Simple dependency injection
BlenderConnectionDep = get_blender_connection
