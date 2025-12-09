# AI System Configuration
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for individual agents"""

    enabled: bool = True
    max_retries: int = 3
    timeout: int = 300  # seconds
    memory_limit: str = "8GB"
    gpu_required: bool = False


class SystemConfig(BaseModel):
    """Main system configuration"""

    # Agent configurations
    agents: Dict[str, AgentConfig] = {
        "scene_composer": AgentConfig(gpu_required=False),
        "asset_generator": AgentConfig(gpu_required=True),
        "terrain_engineer": AgentConfig(gpu_required=True, memory_limit="16GB"),
        "render_controller": AgentConfig(gpu_required=True, memory_limit="12GB"),
        "data_manager": AgentConfig(gpu_required=False),
        "export_specialist": AgentConfig(gpu_required=False),
    }

    # Default paths
    default_output_path: Path = Path("./output")
    default_asset_path: Path = Path("./assets")
    default_render_path: Path = Path("./renders")
    default_export_path: Path = Path("./exports")

    # Performance settings
    max_parallel_jobs: int = 4
    default_complexity: str = "medium"
    default_quality: str = "medium"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # AI Backend settings
    ai_model: str = "huggingface"  # "huggingface", "ollama", "openai"

    # HuggingFace Inference API
    hf_model_id: str = "openai/gpt-oss-20b"  # Default for HF Inference Providers
    hf_provider: Optional[str] = None  # auto or: cerebras, together, nebius, groq, novita

    # Ollama (local)
    ollama_model: str = "qwen3:4b"

    # OpenAI
    openai_model: str = "gpt-4o-mini"

    # Common AI settings
    max_tokens: int = 4000
    temperature: float = 0.7

    # Scene defaults
    default_scene_types: List[str] = [
        "forest",
        "desert",
        "mountain",
        "canyon",
        "coast",
        "kitchen",
        "living_room",
        "bedroom",
        "bathroom",
    ]

    # Asset defaults
    default_asset_types: Dict[str, List[str]] = {
        "creatures": ["carnivore", "herbivore", "bird", "fish"],
        "trees": ["pine", "oak", "palm", "bamboo"],
        "materials": ["ground", "water", "rock", "snow", "sand"],
        "objects": ["rock", "cloud", "particle_system"],
    }

    # Terrain defaults
    default_terrain_types: List[str] = [
        "mountain",
        "canyon",
        "cliff",
        "mesa",
        "river",
        "volcano",
        "coast",
        "multi_mountains",
        "plain",
    ]

    # Render defaults
    default_render_settings: Dict[str, Any] = {
        "resolution": (1280, 720),
        "samples": 128,
        "engine": "cycles",
        "fps": 24,
    }

    # Export defaults
    default_export_formats: List[str] = ["obj", "fbx", "usdc"]
    default_gt_types: List[str] = ["depth", "normal", "segmentation"]

    # Job defaults
    default_tasks: List[str] = [
        "coarse",
        "populate",
        "fine_terrain",
        "render",
        "ground_truth",
    ]

    # Complexity levels
    complexity_levels: Dict[str, Dict[str, Any]] = {
        "low": {
            "detail_level": 0.5,
            "poly_count": "low",
            "texture_resolution": 512,
            "samples": 32,
            "memory_usage": "2-4GB",
        },
        "medium": {
            "detail_level": 1.0,
            "poly_count": "medium",
            "texture_resolution": 1024,
            "samples": 128,
            "memory_usage": "4-8GB",
        },
        "high": {
            "detail_level": 1.5,
            "poly_count": "high",
            "texture_resolution": 2048,
            "samples": 512,
            "memory_usage": "8-16GB",
        },
    }

    # Quality levels
    quality_levels: Dict[str, Dict[str, Any]] = {
        "low": {
            "resolution": (640, 360),
            "samples": 32,
            "denoising": True,
            "motion_blur": False,
        },
        "medium": {
            "resolution": (1280, 720),
            "samples": 128,
            "denoising": True,
            "motion_blur": True,
        },
        "high": {
            "resolution": (1920, 1080),
            "samples": 512,
            "denoising": False,
            "motion_blur": True,
        },
    }


# Global configuration instance
config = SystemConfig()


def get_agent_config(agent_name: str) -> AgentConfig:
    """Get configuration for a specific agent"""
    return config.agents.get(agent_name, AgentConfig())


def get_complexity_config(complexity: str) -> Dict[str, Any]:
    """Get configuration for a complexity level"""
    return config.complexity_levels.get(complexity, config.complexity_levels["medium"])


def get_quality_config(quality: str) -> Dict[str, Any]:
    """Get configuration for a quality level"""
    return config.quality_levels.get(quality, config.quality_levels["medium"])


def update_config(**kwargs) -> None:
    """Update system configuration"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def save_config(config_path: Path) -> None:
    """Save configuration to file"""
    import json

    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)


def load_config(config_path: Path) -> None:
    """Load configuration from file"""
    import json

    with open(config_path, "r") as f:
        config_data = json.load(f)
    global config
    config = SystemConfig(**config_data)
