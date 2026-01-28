"""Model Factory for pydantic_ai agents.

Provides configurable AI model backends:
- HuggingFace Inference API (default)
- Ollama (local)
- OpenAI
"""
import os
from typing import Union

from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

from .ai_system_config import SystemConfig, config


def get_model(cfg: SystemConfig = None) -> Union[str, HuggingFaceModel]:
    """Get the configured AI model for agents.
    
    Args:
        cfg: SystemConfig instance. Uses global config if not provided.
        
    Returns:
        Model identifier string or HuggingFaceModel instance.
    """
    if cfg is None:
        cfg = config
    
    if cfg.ai_model == "huggingface":
        # HuggingFace Inference API
        provider_kwargs = {
            "api_key": os.environ.get("HF_TOKEN"),
        }
        if cfg.hf_provider:
            provider_kwargs["provider_name"] = cfg.hf_provider
            
        return HuggingFaceModel(
            cfg.hf_model_id,
            provider=HuggingFaceProvider(**provider_kwargs)
        )
    
    elif cfg.ai_model == "ollama":
        # Ollama local model
        return f"ollama:{cfg.ollama_model}"
    
    else:
        # OpenAI (default fallback)
        return f"openai:{cfg.openai_model}"


def get_model_string(cfg: SystemConfig = None) -> str:
    """Get model identifier as string for logging/display.
    
    Args:
        cfg: SystemConfig instance. Uses global config if not provided.
        
    Returns:
        Human-readable model identifier string.
    """
    if cfg is None:
        cfg = config
        
    if cfg.ai_model == "huggingface":
        provider = cfg.hf_provider or "auto"
        return f"huggingface:{cfg.hf_model_id} (provider: {provider})"
    elif cfg.ai_model == "ollama":
        return f"ollama:{cfg.ollama_model}"
    else:
        return f"openai:{cfg.openai_model}"
