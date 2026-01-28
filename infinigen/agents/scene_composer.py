# Scene Composer Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent

from infinigen.agent_deps.blender_deps import BlenderConnectionDep
from infinigen.agent_deps.core_deps import SceneInfoManagerDep, SeedManagerDep, ValidationManagerDep
from infinigen.agent_config.model_factory import get_model
from infinigen.agent_tools.blender_tools import BlenderOpsDep
from infinigen.agent_tools.file_tools import FileManagerDep, LoggerDep

logger = logging.getLogger(__name__)


class SceneComposerAgent(BaseModel):
    """Agent specialized in scene composition and orchestration"""

    def __init__(self, **data):
        super().__init__(**data)

        # Agent configuration mit lokalem Modell
        
        self.agent = Agent(
            get_model(),
            result_type=Dict[str, Any],
            system_prompt="""You are a specialized scene composition agent for Infinigen.
            
            Your responsibilities:
            - Compose nature and indoor scenes
            - Manage scene parameters and configuration
            - Validate scene setups
            - Coordinate scene generation workflows
            
            Available scene types:
            - Nature: forest, river, desert, coral_reef, cave, mountain, canyon, plain, cliff, coast, arctic, snowy_mountain
            - Indoor: kitchen, bedroom, living_room, closet, hallway, bathroom, garage, balcony, dining_room
            
            Always validate scene folders and check for required files before proceeding.
            Provide clear feedback on scene composition success/failure.
            """,
        )

    def compose_nature_scene(
        self,
        output_folder: Path,
        scene_seed: int,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
        seed_manager: SeedManagerDep,
        scene_info_manager: SceneInfoManagerDep,
        validation_manager: ValidationManagerDep,
        scene_type: str = "forest",
    ) -> Dict[str, Any]:
        """Compose a nature scene with AI assistance"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "error": "Blender not connected",
                }

            # Set seed for reproducible generation
            seed_manager.set_seed(scene_seed)

            # Validate output folder
            folder_validation = validation_manager.validate_path(
                output_folder, must_exist=False
            )
            if not folder_validation["success"]:
                return folder_validation

            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)

            # Create scene info using SceneInfoManager
            scene_info_result = scene_info_manager.create_scene_info(
                scene_type=scene_type,
                scene_seed=scene_seed,
                output_folder=output_folder,
                status="composed",
                scene_category="nature",
            )

            if not scene_info_result["success"]:
                return scene_info_result

            # Save scene info
            scene_info = scene_info_result["scene_info"]
            save_success = scene_info_manager.save_scene_info(scene_info, output_folder)

            if not save_success:
                return {"success": False, "error": "Failed to save scene info"}

            # Also save as JSON for compatibility
            scene_file = output_folder / f"scene_{scene_seed}_metadata.json"
            success = file_manager.save_json(scene_info, scene_file)

            if success:
                logger_tool.info(
                    f"Successfully composed {scene_type} scene with seed {scene_seed}"
                )
                return {
                    "success": True,
                    "scene_type": scene_type,
                    "scene_seed": scene_seed,
                    "output_folder": str(output_folder),
                    "scene_file": str(scene_file),
                }
            else:
                return {"success": False, "error": "Failed to save scene metadata"}

        except Exception as e:
            logger.error(f"Scene composition failed: {e}")
            return {"success": False, "error": str(e)}

    def compose_indoor_scene(
        self,
        output_folder: Path,
        scene_seed: int,
        blender_ops: BlenderOpsDep,
        blender_conn: BlenderConnectionDep,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
        seed_manager: SeedManagerDep,
        scene_info_manager: SceneInfoManagerDep,
        validation_manager: ValidationManagerDep,
        room_types: List[str] = ["kitchen", "living_room"],
    ) -> Dict[str, Any]:
        """Compose an indoor scene with AI assistance"""
        try:
            if not blender_conn.is_connected:
                return {
                    "success": False,
                    "error": "Blender not connected",
                }

            # Set seed for reproducible generation
            seed_manager.set_seed(scene_seed)

            # Validate output folder
            folder_validation = validation_manager.validate_path(
                output_folder, must_exist=False
            )
            if not folder_validation["success"]:
                return folder_validation

            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)

            # Create scene info using SceneInfoManager
            scene_info_result = scene_info_manager.create_scene_info(
                scene_type="indoor",
                scene_seed=scene_seed,
                output_folder=output_folder,
                room_types=room_types,
                status="composed",
                scene_category="indoor",
            )

            if not scene_info_result["success"]:
                return scene_info_result

            # Save scene info
            scene_info = scene_info_result["scene_info"]
            save_success = scene_info_manager.save_scene_info(scene_info, output_folder)

            if not save_success:
                return {"success": False, "error": "Failed to save scene info"}

            # Also save as JSON for compatibility
            scene_file = output_folder / f"indoor_scene_{scene_seed}_metadata.json"
            success = file_manager.save_json(scene_info, scene_file)

            if success:
                logger_tool.info(
                    f"Successfully composed indoor scene with rooms: {room_types}"
                )
                return {
                    "success": True,
                    "room_types": room_types,
                    "scene_seed": scene_seed,
                    "output_folder": str(output_folder),
                    "scene_file": str(scene_file),
                }
            else:
                return {"success": False, "error": "Failed to save scene metadata"}

        except Exception as e:
            logger.error(f"Indoor scene composition failed: {e}")
            return {"success": False, "error": str(e)}

    def get_scene_recommendations(
        self, scene_type: str, complexity: str = "medium"
    ) -> Dict[str, Any]:
        """Get AI-powered scene composition recommendations"""
        try:
            recommendations = {
                "scene_type": scene_type,
                "complexity": complexity,
                "recommended_seeds": [42, 123, 456, 789, 999],
                "suggested_assets": self._get_asset_recommendations(scene_type),
                "camera_setup": self._get_camera_recommendations(scene_type),
                "lighting_setup": self._get_lighting_recommendations(scene_type),
            }

            return {"success": True, "recommendations": recommendations}

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {"success": False, "error": str(e)}

    def _get_asset_recommendations(self, scene_type: str) -> List[str]:
        """Get asset recommendations based on scene type"""
        asset_map = {
            "forest": ["pine_trees", "oak_trees", "grass", "rocks", "leaves"],
            "desert": ["cactus", "sand", "rocks", "dunes"],
            "mountain": ["pine_trees", "rocks", "snow", "grass"],
            "kitchen": ["fridge", "stove", "sink", "cabinets", "table"],
            "living_room": ["sofa", "tv", "coffee_table", "lamp", "bookshelf"],
        }
        return asset_map.get(scene_type, [])

    def _get_camera_recommendations(self, scene_type: str) -> Dict[str, Any]:
        """Get camera setup recommendations"""
        camera_map = {
            "forest": {"trajectory": "circular", "height": 1.8, "distance": 10},
            "desert": {"trajectory": "linear", "height": 2.0, "distance": 15},
            "mountain": {"trajectory": "spiral", "height": 3.0, "distance": 20},
            "kitchen": {"trajectory": "static", "height": 1.6, "distance": 5},
            "living_room": {"trajectory": "circular", "height": 1.7, "distance": 8},
        }
        return camera_map.get(
            scene_type, {"trajectory": "circular", "height": 1.8, "distance": 10}
        )

    def _get_lighting_recommendations(self, scene_type: str) -> Dict[str, Any]:
        """Get lighting setup recommendations"""
        lighting_map = {
            "forest": {"sun_angle": 45, "sun_strength": 1.0, "ambient": 0.3},
            "desert": {"sun_angle": 60, "sun_strength": 1.5, "ambient": 0.2},
            "mountain": {"sun_angle": 30, "sun_strength": 0.8, "ambient": 0.4},
            "kitchen": {"sun_angle": 90, "sun_strength": 0.5, "ambient": 0.6},
            "living_room": {"sun_angle": 45, "sun_strength": 0.7, "ambient": 0.5},
        }
        return lighting_map.get(
            scene_type, {"sun_angle": 45, "sun_strength": 1.0, "ambient": 0.3}
        )
