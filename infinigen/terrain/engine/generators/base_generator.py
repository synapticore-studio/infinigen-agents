#!/usr/bin/env python3
"""
Base Terrain Generator
Abstract base class for all terrain generators
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class GeneratorConfig:
    """Base configuration for terrain generators"""

    terrain_type: str = "mountain"
    resolution: int = 256
    seed: int = 42
    bounds: tuple = (-50, 50, -50, 50, 0, 100)
    device: str = "cpu"


class BaseTerrainGenerator(ABC):
    """Abstract base class for terrain generators"""

    def __init__(self, config: GeneratorConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate terrain data - must be implemented by subclasses"""
        pass

    def _set_random_seed(self):
        """Set random seed for reproducible results"""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _validate_input(self, data: Any) -> bool:
        """Validate input data - can be overridden by subclasses"""
        return data is not None

    def _log_generation(self, generator_type: str, success: bool, **kwargs):
        """Log generation results"""
        if success:
            self.logger.info(f"✅ {generator_type} generation successful")
        else:
            self.logger.error(f"❌ {generator_type} generation failed")
