# Model Dependencies - Unified model provider using config/model_factory
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

from infinigen.agent_config.model_factory import get_model, get_model_string
from infinigen.agent_config.ai_system_config import config

logger = logging.getLogger(__name__)


@dataclass
class ModelProvider:
    """Unified model provider for all agents.
    
    Uses config/model_factory.py to get the configured model:
    - HuggingFace Inference API
    - Ollama (local)
    - OpenAI
    """
    
    model: Union[str, HuggingFaceModel, None] = None
    model_string: str = ""
    
    def __post_init__(self):
        """Initialize the model from infinigen.agent_config."""
        try:
            self.model = get_model()
            self.model_string = get_model_string()
            logger.info(f"Model provider initialized: {self.model_string}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None
            self.model_string = "none"
    
    def create_agent(
        self,
        result_type: Any = None,
        system_prompt: str = "",
        **kwargs
    ) -> Optional[Agent]:
        """Create a pydantic_ai Agent with the configured model.
        
        Args:
            result_type: The expected result type for the agent.
            system_prompt: System prompt for the agent.
            **kwargs: Additional arguments for the Agent.
            
        Returns:
            Configured Agent instance or None if model unavailable.
        """
        if self.model is None:
            logger.warning("No model available, cannot create agent")
            return None
            
        try:
            agent_kwargs = {
                "model": self.model,
                "system_prompt": system_prompt,
            }
            if result_type is not None:
                agent_kwargs["result_type"] = result_type
            agent_kwargs.update(kwargs)
            
            return Agent(**agent_kwargs)
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None


def get_model_provider() -> ModelProvider:
    """Get a ModelProvider instance."""
    return ModelProvider()


# Dependency for Agents - callable that returns ModelProvider
ModelProviderDep = get_model_provider
