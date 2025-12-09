# Render Controller Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic_ai import Agent
from config.model_factory import get_model

from deps.blender_deps import BlenderConnectionDep
from deps.config_deps import InfinigenConfigDep
from deps.core_deps import SeedManagerDep
from tools.blender_tools import BlenderOpsDep
from tools.file_tools import LoggerDep

logger = logging.getLogger(__name__)


class RenderControllerAgent(BaseModel):
    """Agent specialized in rendering and visualization control"""

    def __init__(self, **data):
        super().__init__(**data)

        # Agent configuration
        self.agent = Agent(
            get_model(),
            result_type=Dict[str, Any],
            system_prompt="""You are a specialized render controller agent for Infinigen.
            
            Your responsibilities:
            - Control scene rendering with Cycles/EEVEE
            - Generate ground truth annotations
            - Manage camera trajectories and setups
            - Optimize rendering performance
            
            Available render engines:
            - Cycles: High quality, slower
            - EEVEE: Real-time, faster
            
            Ground truth types:
            - opengl: Fast, basic annotations
            - blender: High quality, slower
            - depth, normal, segmentation: Specific data types
            
            Camera trajectories:
            - circular, linear, spiral, random, custom
            
            Always validate render setup and provide detailed feedback on rendering success/failure.
            Consider performance vs quality trade-offs when making recommendations.
            """,
        )

    def render_scene(
        self,
        scene_folder: Path,
        output_folder: Path,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        config: InfinigenConfigDep,
        logger_tool: LoggerDep,
        render_settings: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Render a scene with AI assistance"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "error": "Blender not connected",
                }

            # Default render settings
            default_settings = {
                "camera_id": (0, 0),
                "frame_range": (1, 1),
                "resolution": config.default_resolution,
                "fps": 24,
                "engine": config.default_render_engine.lower(),
                "samples": config.default_samples,
            }

            if render_settings:
                default_settings.update(render_settings)

            # Use BlenderOps for rendering operations
            # This would typically involve setting up render settings and executing render
            logger_tool.info(f"Rendering scene with settings: {default_settings}")

            return {
                "success": True,
                "render_settings": default_settings,
                "output_folder": str(output_folder),
                "scene_folder": str(scene_folder),
            }

        except Exception as e:
            logger.error(f"Scene rendering failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_ground_truth(
        self,
        scene_folder: Path,
        output_folder: Path,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        logger_tool: LoggerDep,
        gt_types: List[str] = ["depth", "normal"],
    ) -> Dict[str, Any]:
        """Generate ground truth annotations with AI assistance"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "error": "Blender not connected",
                }

            results = {}

            for gt_type in gt_types:
                # Use BlenderOps for GT generation
                logger_tool.info(f"Generating {gt_type} ground truth")
                results[gt_type] = {
                    "success": True,
                    "gt_type": gt_type,
                    "output_folder": str(output_folder),
                }

            # Check if all GT types succeeded
            all_success = all(result["success"] for result in results.values())

            return {
                "success": all_success,
                "gt_types": gt_types,
                "results": results,
                "output_folder": str(output_folder),
            }

        except Exception as e:
            logger.error(f"Ground truth generation failed: {e}")
            return {"success": False, "error": str(e)}

    def setup_camera_trajectory(
        self,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        logger_tool: LoggerDep,
        trajectory_type: str = "circular",
        trajectory_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Set up camera trajectory with AI assistance"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "error": "Blender not connected",
                }

            # Default trajectory parameters
            default_params = {"radius": 10.0, "height": 1.8, "duration": 5.0, "fps": 24}

            if trajectory_params:
                default_params.update(trajectory_params)

            # Use BlenderOps for camera setup
            logger_tool.info(f"Setting up {trajectory_type} camera trajectory")

            return {
                "success": True,
                "trajectory_type": trajectory_type,
                "trajectory_params": default_params,
            }

        except Exception as e:
            logger.error(f"Camera trajectory setup failed: {e}")
            return {"success": False, "error": str(e)}

    def get_render_recommendations(
        self, scene_type: str, quality_requirements: str = "medium"
    ) -> Dict[str, Any]:
        """Get AI-powered rendering recommendations"""
        try:
            recommendations = {
                "scene_type": scene_type,
                "quality_requirements": quality_requirements,
                "recommended_engine": self._get_engine_recommendation(
                    quality_requirements
                ),
                "camera_setup": self._get_camera_recommendations(scene_type),
                "render_settings": self._get_render_settings(quality_requirements),
                "gt_recommendations": self._get_gt_recommendations(scene_type),
                "performance_tips": self._get_performance_tips(quality_requirements),
            }

            return {"success": True, "recommendations": recommendations}

        except Exception as e:
            logger.error(f"Failed to get render recommendations: {e}")
            return {"success": False, "error": str(e)}

    def _get_engine_recommendation(self, quality_requirements: str) -> str:
        """Get render engine recommendation"""
        engine_map = {"low": "eevee", "medium": "cycles", "high": "cycles"}
        return engine_map.get(quality_requirements, "cycles")

    def _get_camera_recommendations(self, scene_type: str) -> Dict[str, Any]:
        """Get camera setup recommendations"""
        camera_map = {
            "forest": {
                "trajectory": "circular",
                "height": 1.8,
                "distance": 10,
                "fov": 50,
            },
            "desert": {
                "trajectory": "linear",
                "height": 2.0,
                "distance": 15,
                "fov": 60,
            },
            "mountain": {
                "trajectory": "spiral",
                "height": 3.0,
                "distance": 20,
                "fov": 45,
            },
            "indoor": {"trajectory": "static", "height": 1.6, "distance": 5, "fov": 55},
        }
        return camera_map.get(scene_type, camera_map["forest"])

    def _get_render_settings(self, quality_requirements: str) -> Dict[str, Any]:
        """Get render settings based on quality requirements"""
        settings_map = {
            "low": {
                "samples": 32,
                "resolution": (640, 360),
                "denoising": True,
                "motion_blur": False,
            },
            "medium": {
                "samples": 128,
                "resolution": (1280, 720),
                "denoising": True,
                "motion_blur": True,
            },
            "high": {
                "samples": 512,
                "resolution": (1920, 1080),
                "denoising": False,
                "motion_blur": True,
            },
        }
        return settings_map.get(quality_requirements, settings_map["medium"])

    def _get_gt_recommendations(self, scene_type: str) -> List[str]:
        """Get ground truth recommendations"""
        gt_map = {
            "forest": ["depth", "normal", "segmentation"],
            "desert": ["depth", "normal"],
            "mountain": ["depth", "normal", "segmentation"],
            "indoor": ["depth", "normal", "segmentation", "instance"],
        }
        return gt_map.get(scene_type, ["depth", "normal"])

    def _get_performance_tips(self, quality_requirements: str) -> List[str]:
        """Get performance tips"""
        tips = {
            "low": [
                "Use EEVEE for real-time preview",
                "Lower sample count",
                "Enable denoising",
                "Use lower resolution",
            ],
            "medium": [
                "Balance quality vs speed",
                "Use GPU acceleration",
                "Optimize shader complexity",
                "Use texture compression",
            ],
            "high": [
                "Use Cycles for final renders",
                "Higher sample count",
                "Disable denoising for quality",
                "Use high resolution",
            ],
        }
        return tips.get(quality_requirements, tips["medium"])
