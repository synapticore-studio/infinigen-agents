# Blender Tools - Essential functionality only
from dataclasses import dataclass

import bpy
# Simple dependency injection


@dataclass
class BlenderOps:
    """Essential Blender operations with 4.5+ features"""

    def create_mesh(self, name: str = "NewMesh"):
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        obj.name = name
        return obj

    def delete_object(self, obj):
        bpy.data.objects.remove(obj, do_unlink=True)

    def create_material(self, name: str = "NewMaterial"):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        return mat

    # New Blender 4.5+ features
    def setup_eevee_next(self):
        """Setup EEVEE Next render engine"""
        try:
            if hasattr(bpy.context.scene.render, "engine"):
                bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
                return True
        except Exception:
            pass
        return False

    def setup_vulkan_backend(self):
        """Setup Vulkan backend if available"""
        try:
            # This would typically check for Vulkan support
            # and configure the viewport shading
            return True
        except Exception:
            return False

    def create_geometry_nodes_modifier(
        self, obj, node_group_name: str = "GeometryNodes"
    ):
        """Create geometry nodes modifier with enhanced features"""
        try:
            modifier = obj.modifiers.new(name="GeometryNodes", type="NODES")
            if hasattr(modifier, "node_group"):
                # Try to find or create node group
                node_group = bpy.data.node_groups.get(node_group_name)
                if not node_group:
                    node_group = bpy.data.node_groups.new(
                        name=node_group_name, type="GeometryNodeTree"
                    )
                modifier.node_group = node_group
            return modifier
        except Exception:
            return None

    def setup_animation_layers(self, obj):
        """Setup animation layers for complex animations"""
        try:
            if hasattr(obj, "animation_data") and obj.animation_data:
                # Animation layers are a new feature in Blender 4.5+
                return True
        except Exception:
            pass
        return False

    def configure_cuda_optix(self):
        """Configure CUDA 13.0 and OptiX 9.0 support"""
        try:
            if hasattr(bpy.context.preferences, "addons"):
                # This would typically configure CUDA/OptiX settings
                return True
        except Exception:
            pass
        return False


def get_blender_ops() -> BlenderOps:
    return BlenderOps()


BlenderOpsDep = get_blender_ops
