# Configuration Dependencies - Pure data only
from dataclasses import dataclass
from pathlib import Path

# Simple dependency injection


@dataclass
class InfinigenConfig:
    """Infinigen configuration - pure data only"""

    base_path: Path = Path(".")
    output_path: Path = Path("./output")
    default_scene_type: str = "nature"
    ai_model: str = "local"  # Lokale distilierte Modelle
    local_model_path: str = "./models/infinigen-distilled"
    model_type: str = "ollama"
    ollama_model: str = "infinigen:latest"
    max_tokens: int = 4000
    temperature: float = 0.7


def get_infinigen_config() -> InfinigenConfig:
    return InfinigenConfig()


InfinigenConfigDep = get_infinigen_config
