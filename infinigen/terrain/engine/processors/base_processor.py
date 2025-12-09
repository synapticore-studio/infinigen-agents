#!/usr/bin/env python3
"""
Base Terrain Processor
Abstract base class for all terrain processors
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class ProcessorConfig:
    """Base configuration for terrain processors"""

    terrain_type: str = "mountain"
    resolution: int = 256
    seed: int = 42
    device: str = "cpu"


class BaseTerrainProcessor(ABC):
    """Abstract base class for terrain processors"""

    def __init__(self, config: ProcessorConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process_height_map(self, height_map: np.ndarray) -> np.ndarray:
        """Process height map - must be implemented by subclasses"""
        pass

    def _set_random_seed(self):
        """Set random seed for reproducible results"""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _validate_input(self, data: Any) -> bool:
        """Validate input data - can be overridden by subclasses"""
        return data is not None

    def _log_processing(self, processor_type: str, success: bool, **kwargs):
        """Log processing results"""
        if success:
            self.logger.info(f"✅ {processor_type} processing successful")
        else:
            self.logger.error(f"❌ {processor_type} processing failed")
