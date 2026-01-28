# Core Tools - Essential Infinigen integration tools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Simple dependency injection

from infinigen.agent_deps.core_deps import (
    GinManagerDep,
    MathUtilsDep,
    SceneInfoManagerDep,
    SeedManagerDep,
    TaskManagerDep,
    ValidationManagerDep,
)


@dataclass
class CoreOrchestrator:
    """Essential core orchestration functionality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_system_setup(
        self,
        seed_manager: SeedManagerDep,
        task_manager: TaskManagerDep,
        validation_manager: ValidationManagerDep,
    ) -> Dict[str, Any]:
        """Validate that the core system is properly set up"""
        try:
            # Validate seed manager
            seed_manager.set_seed(42)

            # Validate task manager
            task_validation = task_manager.validate_tasks(["coarse", "populate"])

            # Validate validation manager
            test_path = Path(".")
            path_validation = validation_manager.validate_path(
                test_path, must_exist=True
            )

            return {
                "success": True,
                "seed_manager_ready": True,
                "task_manager_ready": task_validation["success"],
                "validation_manager_ready": path_validation["success"],
                "system_ready": True,
            }
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return {"success": False, "error": str(e)}

    def get_system_status(
        self,
        seed_manager: SeedManagerDep,
        task_manager: TaskManagerDep,
        gin_manager: GinManagerDep,
        scene_info_manager: SceneInfoManagerDep,
        math_utils: MathUtilsDep,
        validation_manager: ValidationManagerDep,
    ) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "success": True,
                "seed_manager": {
                    "base_seed": seed_manager.base_seed,
                    "current_seed": seed_manager.current_seed,
                },
                "task_manager": {
                    "available_tasks": task_manager.available_tasks,
                },
                "gin_manager": {
                    "config_folders": [str(f) for f in gin_manager.config_folders],
                },
                "scene_info_manager": {
                    "ready": True,
                },
                "math_utils": {
                    "ready": True,
                },
                "validation_manager": {
                    "ready": True,
                },
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"success": False, "error": str(e)}

    def create_workflow_plan(
        self,
        task_manager: TaskManagerDep,
        requested_tasks: List[str],
        scene_type: str = "forest",
    ) -> Dict[str, Any]:
        """Create a workflow plan for task execution"""
        try:
            # Validate tasks
            task_validation = task_manager.validate_tasks(requested_tasks)
            if not task_validation["success"]:
                return task_validation

            # Create task sequence
            task_sequence = task_manager.create_task_sequence(requested_tasks)

            # Get dependencies for each task
            task_dependencies = {}
            for task in task_sequence:
                task_dependencies[task] = task_manager.get_task_dependencies(task)

            return {
                "success": True,
                "scene_type": scene_type,
                "requested_tasks": requested_tasks,
                "task_sequence": task_sequence,
                "task_dependencies": task_dependencies,
                "execution_plan": {
                    "total_tasks": len(task_sequence),
                    "estimated_duration": len(task_sequence) * 5,  # minutes per task
                },
            }
        except Exception as e:
            self.logger.error(f"Failed to create workflow plan: {e}")
            return {"success": False, "error": str(e)}


@dataclass
class SceneCoordinator:
    """Essential scene coordination functionality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def coordinate_scene_creation(
        self,
        scene_type: str,
        scene_seed: int,
        output_folder: Path,
        seed_manager: SeedManagerDep,
        scene_info_manager: SceneInfoManagerDep,
        validation_manager: ValidationManagerDep,
    ) -> Dict[str, Any]:
        """Coordinate the creation of a complete scene"""
        try:
            # Set seed
            seed_manager.set_seed(scene_seed)

            # Validate output folder
            folder_validation = validation_manager.validate_path(
                output_folder, must_exist=False
            )
            if not folder_validation["success"]:
                return folder_validation

            # Create output folder if it doesn't exist
            output_folder.mkdir(parents=True, exist_ok=True)

            # Create scene info
            scene_info_result = scene_info_manager.create_scene_info(
                scene_type=scene_type,
                scene_seed=scene_seed,
                output_folder=output_folder,
            )

            if not scene_info_result["success"]:
                return scene_info_result

            # Save scene info
            scene_info = scene_info_result["scene_info"]
            save_success = scene_info_manager.save_scene_info(scene_info, output_folder)

            if not save_success:
                return {"success": False, "error": "Failed to save scene info"}

            return {
                "success": True,
                "scene_type": scene_type,
                "scene_seed": scene_seed,
                "output_folder": str(output_folder),
                "scene_info": scene_info,
                "ready_for_processing": True,
            }
        except Exception as e:
            self.logger.error(f"Scene coordination failed: {e}")
            return {"success": False, "error": str(e)}


# Factory functions for dependency injection
def get_core_orchestrator() -> CoreOrchestrator:
    return CoreOrchestrator()


def get_scene_coordinator() -> SceneCoordinator:
    return SceneCoordinator()


# Pydantic-AI Dependencies
CoreOrchestratorDep = get_core_orchestrator
SceneCoordinatorDep = get_scene_coordinator
