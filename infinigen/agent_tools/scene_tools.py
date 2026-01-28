# Scene Composition Tools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SceneTools(BaseModel):
    """Tools for scene composition and management"""

    def compose_nature_scene(
        self, output_folder: Path, scene_seed: int, scene_type: str = "forest"
    ) -> Dict[str, Any]:
        """Compose a nature scene with specified parameters"""
        try:
            # Import here to avoid circular dependencies
            from infinigen_examples.generate_nature import compose_nature

            result = compose_nature(output_folder=output_folder, scene_seed=scene_seed)

            logger.info(f"Composed nature scene: {scene_type} with seed {scene_seed}")
            return {
                "success": True,
                "scene_type": scene_type,
                "seed": scene_seed,
                "output_folder": str(output_folder),
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to compose nature scene: {e}")
            return {"success": False, "error": str(e)}

    def compose_indoor_scene(
        self,
        output_folder: Path,
        scene_seed: int,
        room_types: List[str] = ["kitchen", "living_room"],
    ) -> Dict[str, Any]:
        """Compose an indoor scene with specified room types"""
        try:
            from infinigen_examples.generate_indoors import compose_indoors

            result = compose_indoors(output_folder=output_folder, scene_seed=scene_seed)

            logger.info(f"Composed indoor scene with rooms: {room_types}")
            return {
                "success": True,
                "room_types": room_types,
                "seed": scene_seed,
                "output_folder": str(output_folder),
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to compose indoor scene: {e}")
            return {"success": False, "error": str(e)}

    def get_scene_info(self, scene_folder: Path) -> Dict[str, Any]:
        """Get information about an existing scene"""
        try:
            info_file = scene_folder / "assets" / "info.pickle"
            if info_file.exists():
                import pickle

                with open(info_file, "rb") as f:
                    info = pickle.load(f)
                return {"success": True, "info": info}
            else:
                return {"success": False, "error": "No scene info found"}
        except Exception as e:
            logger.error(f"Failed to get scene info: {e}")
            return {"success": False, "error": str(e)}

    def list_available_scene_types(self) -> List[str]:
        """List available scene types"""
        return [
            "forest",
            "river",
            "desert",
            "coral_reef",
            "cave",
            "mountain",
            "canyon",
            "plain",
            "cliff",
            "coast",
            "arctic",
            "snowy_mountain",
            "under_water",
            "kelp_forest",
        ]

    def validate_scene_folder(self, scene_folder: Path) -> Dict[str, Any]:
        """Validate that a scene folder contains required files"""
        required_files = ["scene.blend", "assets/info.pickle"]
        missing_files = []

        for file_path in required_files:
            if not (scene_folder / file_path).exists():
                missing_files.append(file_path)

        return {
            "valid": len(missing_files) == 0,
            "missing_files": missing_files,
            "scene_folder": str(scene_folder),
        }
