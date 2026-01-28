# Asset Dependencies - Pure data only
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Simple dependency injection


@dataclass
class AssetParameterConfig:
    """Asset parameter configuration - pure data only"""

    # Asset categories
    asset_categories: List[str] = None
    factory_types: List[str] = None

    # Parameter settings
    global_params: Dict[str, Any] = None
    individual_params: List[Dict[str, Any]] = None

    # Generation settings
    repeat_count: int = 12
    scene_idx: int = 0
    spawn_placeholder: bool = False

    # Quality settings
    detail_level: str = "medium"  # low, medium, high
    face_size: float = 0.1
    distance_threshold: float = 10.0

    def __post_init__(self):
        if self.asset_categories is None:
            self.asset_categories = [
                "creatures",
                "trees",
                "materials",
                "objects",
                "furniture",
                "appliances",
                "decorations",
            ]

        if self.factory_types is None:
            self.factory_types = [
                "ChairFactory",
                "PanFactory",
                "PotFactory",
                "FruitContainerFactory",
                "CarnivoreFactory",
                "HerbivoreFactory",
                "PineTreeFactory",
                "OakTreeFactory",
            ]

        if self.global_params is None:
            self.global_params = {}

        if self.individual_params is None:
            self.individual_params = [{}]


@dataclass
class AssetFactoryConfig:
    """Asset factory configuration - pure data only"""

    # Factory settings
    factory_seed: Optional[int] = None
    coarse_mode: bool = False

    # Spawn settings
    spawn_location: tuple = (0, 0, 0)
    spawn_rotation: tuple = (0, 0, 0)
    spawn_scale: tuple = (1, 1, 1)

    # Export settings
    export_enabled: bool = False
    export_path: Optional[str] = None
    semantic_mapping: bool = True

    # Performance settings
    garbage_collection: bool = True
    verbose_logging: bool = False


def get_asset_parameter_config() -> AssetParameterConfig:
    return AssetParameterConfig()


def get_asset_factory_config() -> AssetFactoryConfig:
    return AssetFactoryConfig()


AssetParameterConfigDep = get_asset_parameter_config
AssetFactoryConfigDep = get_asset_factory_config
