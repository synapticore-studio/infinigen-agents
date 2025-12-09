# Constraint Dependencies - Pure data only
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Simple dependency injection


@dataclass
class ConstraintConfig:
    """Constraint system configuration - pure data only"""

    # Constraint types
    constraint_types: List[str] = None
    room_constraints_enabled: bool = True
    furniture_constraints_enabled: bool = True
    spatial_constraints_enabled: bool = True

    # Solver settings
    solver_timeout: int = 300  # seconds
    max_iterations: int = 1000
    tolerance: float = 0.01

    # Room settings
    room_types: List[str] = None
    furniture_fullness_pct: float = 0.75
    obj_interior_obj_pct: float = 0.7
    obj_on_storage_pct: float = 0.75
    obj_on_nonstorage_pct: float = 0.6

    def __post_init__(self):
        if self.constraint_types is None:
            self.constraint_types = ["room", "furniture", "spatial", "semantic"]

        if self.room_types is None:
            self.room_types = [
                "bedroom",
                "living_room",
                "kitchen",
                "bathroom",
                "dining_room",
                "office",
                "garage",
                "balcony",
            ]


@dataclass
class GinConfig:
    """Gin configuration system - pure data only"""

    # Config paths
    config_folders: List[str] = None
    base_config: str = "base.gin"
    scene_type: str = "forest"

    # Gin settings
    skip_unknown: bool = True
    finalize_config: bool = False

    # Overrides
    overrides: List[str] = None

    def __post_init__(self):
        if self.config_folders is None:
            self.config_folders = [
                "infinigen_examples/configs_nature",
                "infinigen_examples/configs_indoor",
            ]

        if self.overrides is None:
            self.overrides = []


def get_constraint_config() -> ConstraintConfig:
    return ConstraintConfig()


def get_gin_config() -> GinConfig:
    return GinConfig()


ConstraintConfigDep = get_constraint_config
GinConfigDep = get_gin_config
