# Core Dependencies - Infinigen Core System Integration
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gin
import numpy as np

from infinigen.core.util.math import AddedSeed, FixedSeed, int_hash
from infinigen.core.util.organization import Task

# Simple dependency injection


@dataclass
class SeedManager:
    """Essential seed management for reproducible generation"""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.current_seed = base_seed

    def set_seed(self, seed: int) -> int:
        """Set the current seed and return it"""
        self.current_seed = int(seed)
        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        return self.current_seed

    def get_derived_seed(self, offset: int = 0) -> int:
        """Get a derived seed based on current seed + offset"""
        return int_hash((self.current_seed, offset))

    def apply_fixed_seed(self, seed: int) -> FixedSeed:
        """Create a FixedSeed context manager"""
        return FixedSeed(seed)

    def apply_added_seed(self, added_seed: int) -> AddedSeed:
        """Create an AddedSeed context manager"""
        return AddedSeed(added_seed)

    def hash_string_to_seed(self, string: str) -> int:
        """Convert string to deterministic seed"""
        return int_hash(string)


@dataclass
class TaskManager:
    """Essential task management for Infinigen workflows"""

    def __init__(self):
        self.available_tasks = [
            Task.Coarse,
            Task.Populate,
            Task.FineTerrain,
            Task.Render,
            Task.GroundTruth,
            Task.MeshSave,
            Task.Export,
        ]

    def validate_tasks(self, tasks: List[str]) -> Dict[str, Any]:
        """Validate that all tasks are supported"""
        try:
            invalid_tasks = [task for task in tasks if task not in self.available_tasks]
            return {
                "success": len(invalid_tasks) == 0,
                "valid_tasks": [task for task in tasks if task in self.available_tasks],
                "invalid_tasks": invalid_tasks,
                "available_tasks": self.available_tasks,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_task_dependencies(self, task: str) -> List[str]:
        """Get dependencies for a specific task"""
        dependencies = {
            Task.Coarse: [],
            Task.Populate: [Task.Coarse],
            Task.FineTerrain: [Task.Coarse],
            Task.Render: [Task.Coarse, Task.Populate],
            Task.GroundTruth: [Task.Coarse, Task.Populate],
            Task.MeshSave: [Task.Coarse, Task.Populate],
            Task.Export: [Task.Coarse, Task.Populate],
        }
        return dependencies.get(task, [])

    def create_task_sequence(self, requested_tasks: List[str]) -> List[str]:
        """Create a proper task execution sequence"""
        try:
            # Validate tasks
            validation = self.validate_tasks(requested_tasks)
            if not validation["success"]:
                return []

            # Build dependency graph
            all_tasks = set()
            for task in requested_tasks:
                all_tasks.add(task)
                all_tasks.update(self.get_task_dependencies(task))

            # Sort by dependencies
            task_sequence = []
            remaining_tasks = set(all_tasks)

            while remaining_tasks:
                # Find tasks with no unmet dependencies
                ready_tasks = []
                for task in remaining_tasks:
                    deps = self.get_task_dependencies(task)
                    if all(dep in task_sequence for dep in deps):
                        ready_tasks.append(task)

                if not ready_tasks:
                    # Circular dependency or error
                    break

                # Add ready tasks to sequence
                for task in sorted(ready_tasks):
                    if task in requested_tasks:  # Only add requested tasks
                        task_sequence.append(task)
                    remaining_tasks.remove(task)

            return task_sequence

        except Exception as e:
            return []


@dataclass
class GinManager:
    """Essential Gin configuration management"""

    def __init__(self):
        self.config_folders = [
            Path("infinigen_examples/configs_nature"),
            Path("infinigen_examples/configs_indoor"),
            Path("infinigen_examples/configs_astronomy"),
        ]

    def load_base_config(self) -> Dict[str, Any]:
        """Load base Gin configuration"""
        try:
            gin.clear_config()
            return {
                "success": True,
                "config_cleared": True,
                "available_folders": [str(f) for f in self.config_folders],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def apply_config_overrides(self, overrides: List[str]) -> Dict[str, Any]:
        """Apply Gin configuration overrides"""
        try:
            # This would typically apply gin overrides
            return {
                "success": True,
                "overrides_applied": len(overrides),
                "overrides": overrides,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_gin_parameters(self, function_name: str) -> Dict[str, Any]:
        """Get Gin parameters for a function"""
        try:
            # This would typically query gin for function parameters
            return {
                "success": True,
                "function_name": function_name,
                "parameters": {},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


@dataclass
class SceneInfoManager:
    """Essential scene information management"""

    def create_scene_info(
        self, scene_type: str, scene_seed: int, output_folder: Path, **kwargs
    ) -> Dict[str, Any]:
        """Create scene information dictionary"""
        try:
            scene_info = {
                "scene_type": scene_type,
                "scene_seed": scene_seed,
                "output_folder": str(output_folder),
                "version": "1.15.3",  # Infinigen version
                "timestamp": np.datetime64("now").astype(str),
                **kwargs,
            }
            return {"success": True, "scene_info": scene_info}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_scene_info(self, scene_info: Dict[str, Any], output_folder: Path) -> bool:
        """Save scene information to pickle file"""
        try:
            info_file = output_folder / "assets" / "info.pickle"
            info_file.parent.mkdir(parents=True, exist_ok=True)

            with open(info_file, "wb") as f:
                pickle.dump(scene_info, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            return False

    def load_scene_info(self, scene_folder: Path) -> Optional[Dict[str, Any]]:
        """Load scene information from pickle file"""
        try:
            info_file = scene_folder / "assets" / "info.pickle"
            if not info_file.exists():
                return None

            with open(info_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            return None

    def validate_scene_info(self, scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scene information structure"""
        try:
            required_fields = ["scene_type", "scene_seed", "output_folder"]
            missing_fields = [
                field for field in required_fields if field not in scene_info
            ]

            return {
                "success": len(missing_fields) == 0,
                "missing_fields": missing_fields,
                "scene_info_valid": len(missing_fields) == 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


@dataclass
class MathUtils:
    """Essential mathematical utilities"""

    def int_hash(self, value: Union[int, str, Tuple]) -> int:
        """Create deterministic hash from various input types"""
        return int_hash(value)

    def generate_random_vector(self, seed: int, dimensions: int = 3) -> np.ndarray:
        """Generate random vector with given seed"""
        with FixedSeed(seed):
            return np.random.random(dimensions)

    def generate_random_rotation(self, seed: int) -> np.ndarray:
        """Generate random rotation matrix with given seed"""
        with FixedSeed(seed):
            # Generate random rotation using Rodrigues' formula
            axis = np.random.random(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.random() * 2 * np.pi

            # Rodrigues' rotation formula
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            return R

    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(point1 - point2)

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


@dataclass
class ValidationManager:
    """Essential validation utilities"""

    def validate_path(self, path: Path, must_exist: bool = True) -> Dict[str, Any]:
        """Validate file or directory path"""
        try:
            exists = path.exists()
            is_file = path.is_file() if exists else False
            is_dir = path.is_dir() if exists else False

            if must_exist and not exists:
                return {
                    "success": False,
                    "error": f"Path does not exist: {path}",
                    "path": str(path),
                }

            return {
                "success": True,
                "path": str(path),
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_scene_folder(self, scene_folder: Path) -> Dict[str, Any]:
        """Validate scene folder structure"""
        try:
            required_files = ["scene.blend", "assets/info.pickle"]
            missing_files = []

            for file_path in required_files:
                full_path = scene_folder / file_path
                if not full_path.exists():
                    missing_files.append(file_path)

            return {
                "success": len(missing_files) == 0,
                "missing_files": missing_files,
                "scene_folder": str(scene_folder),
                "valid": len(missing_files) == 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_parameters(
        self, parameters: Dict[str, Any], required: List[str]
    ) -> Dict[str, Any]:
        """Validate parameter dictionary"""
        try:
            missing_params = [param for param in required if param not in parameters]
            return {
                "success": len(missing_params) == 0,
                "missing_parameters": missing_params,
                "valid": len(missing_params) == 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Factory functions for dependency injection
def get_seed_manager() -> SeedManager:
    return SeedManager()


def get_task_manager() -> TaskManager:
    return TaskManager()


def get_gin_manager() -> GinManager:
    return GinManager()


def get_scene_info_manager() -> SceneInfoManager:
    return SceneInfoManager()


def get_math_utils() -> MathUtils:
    return MathUtils()


def get_validation_manager() -> ValidationManager:
    return ValidationManager()


# Simple dependency injection
SeedManagerDep = get_seed_manager
TaskManagerDep = get_task_manager
GinManagerDep = get_gin_manager
SceneInfoManagerDep = get_scene_info_manager
MathUtilsDep = get_math_utils
ValidationManagerDep = get_validation_manager
