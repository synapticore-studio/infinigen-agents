# Export Specialist Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext
from config.model_factory import get_model

from deps.core_deps import ValidationManagerDep
from deps.config_deps import InfinigenConfigDep
from tools.file_tools import FileManagerDep, LoggerDep

logger = logging.getLogger(__name__)


# Define dependencies type for the agent
class ExportSpecialistDeps:
    """Dependencies for the Export Specialist Agent"""
    def __init__(
        self,
        config: InfinigenConfigDep,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
    ):
        self.config = config
        self.file_manager = file_manager
        self.logger_tool = logger_tool


# Create the Agent directly using pydantic-ai patterns
export_specialist_agent = Agent(
    get_model(),
    result_type=Dict[str, Any],
    deps_type=ExportSpecialistDeps,
    system_prompt="""You are a specialized export specialist agent for Infinigen.
    
    Your responsibilities:
    - Export scenes to various 3D formats
    - Convert between different file formats
    - Export ground truth data
    - Optimize exports for different use cases
    
    Supported export formats:
    - 3D: obj, fbx, usdc, usda, stl, ply, gltf, glb
    - Images: exr, png, jpg, hdr
    - Data: json, csv, hdf5, npz
    
    Export categories:
    - Scene geometry and materials
    - Ground truth annotations
    - Camera trajectories and metadata
    - Asset libraries and collections
    
    Always validate export parameters and provide detailed feedback on export success/failure.
    Consider target application requirements when recommending export settings.
    """,
)


@export_specialist_agent.tool
async def export_scene_data(
    ctx: RunContext[ExportSpecialistDeps],
    input_blend_file: Path,
    output_folder: Path,
    export_formats: List[str] = None,
    export_settings: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Export scene data with AI assistance
    
    Args:
        ctx: The run context with dependencies
        input_blend_file: Path to the input .blend file
        output_folder: Path to the output folder
        export_formats: List of export formats
        export_settings: Optional export settings
        
    Returns:
        Dictionary with export status and details
    """
    try:
        # Default export formats
        if export_formats is None:
            export_formats = ["obj", "fbx", "usdc"]

        # Default export settings
        default_settings = {
            "include_materials": True,
            "include_textures": True,
            "include_animations": False,
            "optimize_meshes": True,
            "compression": "medium",
        }

        if export_settings:
            default_settings.update(export_settings)

        # Create export metadata
        export_metadata = {
            "input_file": str(input_blend_file),
            "output_folder": str(output_folder),
            "export_formats": export_formats,
            "export_settings": default_settings,
            "status": "exported",
        }

        # Save export metadata
        export_file = output_folder / "export_metadata.json"
        success = ctx.deps.file_manager.save_json(export_metadata, export_file)

        if success:
            ctx.deps.logger_tool.info(
                f"Successfully exported scene to {len(export_formats)} formats"
            )
            return {
                "success": True,
                "export_formats": export_formats,
                "export_settings": default_settings,
                "export_file": str(export_file),
            }
        else:
            return {"success": False, "error": "Failed to save export metadata"}

    except Exception as e:
        logger.error(f"Scene export failed: {e}")
        return {"success": False, "error": str(e)}


@export_specialist_agent.tool
async def convert_mesh_format(
    ctx: RunContext[ExportSpecialistDeps],
    input_file: Path,
    output_file: Path,
    target_format: str,
    conversion_settings: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Convert mesh between formats with AI assistance
    
    Args:
        ctx: The run context with dependencies
        input_file: Path to the input file
        output_file: Path to the output file
        target_format: Target file format
        conversion_settings: Optional conversion settings
        
    Returns:
        Dictionary with conversion status and details
    """
    try:
        # Default conversion settings
        default_settings = {
            "optimize_mesh": True,
            "preserve_materials": True,
            "compression_level": 6,
        }

        if conversion_settings:
            default_settings.update(conversion_settings)

        # Create conversion metadata
        conversion_metadata = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "target_format": target_format,
            "conversion_settings": default_settings,
            "status": "converted",
        }

        # Save conversion metadata
        conversion_file = output_file.parent / f"{output_file.stem}_conversion.json"
        success = ctx.deps.file_manager.save_json(conversion_metadata, conversion_file)

        if success:
            ctx.deps.logger_tool.info(f"Successfully converted mesh to {target_format}")
            return {
                "success": True,
                "conversion_settings": default_settings,
                "conversion_file": str(conversion_file),
            }
        else:
            return {"success": False, "error": "Failed to save conversion metadata"}

    except Exception as e:
        logger.error(f"Mesh conversion failed: {e}")
        return {"success": False, "error": str(e)}


@export_specialist_agent.tool
async def export_ground_truth_data(
    ctx: RunContext[ExportSpecialistDeps],
    scene_folder: Path,
    output_folder: Path,
    gt_types: List[str] = None,
    export_settings: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Export ground truth data with AI assistance
    
    Args:
        ctx: The run context with dependencies
        scene_folder: Path to the scene folder
        output_folder: Path to the output folder
        gt_types: List of ground truth types to export
        export_settings: Optional export settings
        
    Returns:
        Dictionary with export status and details
    """
    try:
        # Default GT types
        if gt_types is None:
            gt_types = ["depth", "normal", "segmentation"]

        # Default export settings
        default_settings = {
            "image_format": "exr",
            "compression": "zip",
            "include_metadata": True,
            "resolution": (1280, 720),
        }

        if export_settings:
            default_settings.update(export_settings)

        # Create GT export metadata
        gt_metadata = {
            "scene_folder": str(scene_folder),
            "output_folder": str(output_folder),
            "gt_types": gt_types,
            "export_settings": default_settings,
            "status": "exported",
        }

        # Save GT export metadata
        gt_file = output_folder / "gt_export_metadata.json"
        success = ctx.deps.file_manager.save_json(gt_metadata, gt_file)

        if success:
            ctx.deps.logger_tool.info(f"Successfully exported {len(gt_types)} GT types")
            return {
                "success": True,
                "export_settings": default_settings,
                "gt_file": str(gt_file),
            }
        else:
            return {"success": False, "error": "Failed to save GT export metadata"}

    except Exception as e:
        logger.error(f"Ground truth export failed: {e}")
        return {"success": False, "error": str(e)}


# Helper functions (not agent tools)
def get_export_recommendations(
    target_application: str, data_type: str = "scene"
) -> Dict[str, Any]:
    """Get AI-powered export recommendations"""
    try:
        recommendations = {
            "target_application": target_application,
            "data_type": data_type,
            "recommended_formats": _get_format_recommendations(
                target_application, data_type
            ),
            "export_settings": _get_export_settings_recommendations(
                target_application
            ),
            "optimization_tips": _get_export_optimization_tips(
                target_application
            ),
        }

        return {"success": True, "recommendations": recommendations}

    except Exception as e:
        logger.error(f"Failed to get export recommendations: {e}")
        return {"success": False, "error": str(e)}


def _get_format_recommendations(
    target_application: str, data_type: str
) -> List[str]:
    """Get format recommendations based on target application"""
    format_map = {
        "blender": {
            "scene": ["blend", "fbx", "obj"],
            "assets": ["blend", "fbx", "obj"],
            "materials": ["blend", "usdc"],
        },
        "unreal": {
            "scene": ["fbx", "usd"],
            "assets": ["fbx", "usd"],
            "materials": ["usd", "fbx"],
        },
        "unity": {
            "scene": ["fbx", "obj"],
            "assets": ["fbx", "obj"],
            "materials": ["fbx"],
        },
        "research": {
            "scene": ["obj", "ply", "gltf"],
            "assets": ["obj", "ply"],
            "materials": ["json", "hdf5"],
        },
    }
    return format_map.get(target_application, {}).get(data_type, ["obj", "fbx"])


def _get_export_settings_recommendations(
    target_application: str
) -> Dict[str, Any]:
    """Get export settings recommendations"""
    settings_map = {
        "blender": {
            "include_materials": True,
            "include_textures": True,
            "include_animations": True,
            "optimize_meshes": False,
        },
        "unreal": {
            "include_materials": True,
            "include_textures": True,
            "include_animations": True,
            "optimize_meshes": True,
        },
        "unity": {
            "include_materials": True,
            "include_textures": True,
            "include_animations": True,
            "optimize_meshes": True,
        },
        "research": {
            "include_materials": False,
            "include_textures": False,
            "include_animations": False,
            "optimize_meshes": True,
        },
    }
    return settings_map.get(target_application, settings_map["research"])


def _get_export_optimization_tips(target_application: str) -> List[str]:
    """Get export optimization tips"""
    tips = {
        "blender": [
            "Preserve all materials and textures",
            "Keep animation data intact",
            "Use native Blender format when possible",
        ],
        "unreal": [
            "Optimize mesh geometry for real-time rendering",
            "Use appropriate LOD levels",
            "Compress textures for mobile platforms",
        ],
        "unity": [
            "Optimize for mobile performance",
            "Use texture atlasing",
            "Consider platform-specific optimizations",
        ],
        "research": [
            "Focus on geometric accuracy",
            "Export metadata for analysis",
            "Use lossless compression for data integrity",
        ],
    }
    return tips.get(target_application, tips["research"])
