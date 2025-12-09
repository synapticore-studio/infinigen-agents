#!/usr/bin/env python3
"""
Terrain Map Generator
Generates height maps, normal maps, and other terrain maps using modern algorithms
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

# OpenCV import with fallback
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from .base_generator import BaseTerrainGenerator, GeneratorConfig

logger = logging.getLogger(__name__)


class TerrainMapGenerator(BaseTerrainGenerator):
    """Generates various terrain maps using modern algorithms"""

    def __init__(self, config: GeneratorConfig, device: str = "cpu"):
        super().__init__(config, device)
        self._set_random_seed()

    def generate_height_map(self) -> Optional[np.ndarray]:
        """Generate height map based on terrain type"""
        try:
            self.logger.info(f"Generating height map for {self.config.terrain_type}")

            if self.config.terrain_type == "mountain":
                return self._generate_mountain_height_map()
            elif self.config.terrain_type == "hills":
                return self._generate_hills_height_map()
            elif self.config.terrain_type == "valley":
                return self._generate_valley_height_map()
            elif self.config.terrain_type == "plateau":
                return self._generate_plateau_height_map()
            elif self.config.terrain_type == "desert":
                return self._generate_desert_height_map()
            elif self.config.terrain_type == "ocean":
                return self._generate_ocean_height_map()
            else:
                return self._generate_default_height_map()

        except Exception as e:
            self.logger.error(f"Error generating height map: {e}")
            return None

    def generate_normal_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generate normal map from height map"""
        try:
            # Calculate gradients
            grad_x = np.gradient(height_map, axis=1)
            grad_y = np.gradient(height_map, axis=0)

            # Create normal map
            normal_map = np.zeros((*height_map.shape, 3))
            normal_map[:, :, 0] = -grad_x
            normal_map[:, :, 1] = -grad_y
            normal_map[:, :, 2] = 1.0

            # Normalize
            norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
            normal_map = normal_map / (norm + 1e-8)

            return normal_map

        except Exception as e:
            self.logger.error(f"Error generating normal map: {e}")
            return np.zeros((*height_map.shape, 3))

    def generate_displacement_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generate displacement map from height map"""
        try:
            # Apply noise for displacement
            noise = np.random.normal(0, 0.1, height_map.shape)
            displacement_map = height_map + noise

            # Smooth the displacement
            displacement_map = gaussian_filter(displacement_map, sigma=1.0)

            return displacement_map

        except Exception as e:
            self.logger.error(f"Error generating displacement map: {e}")
            return height_map

    def generate_roughness_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generate roughness map from height map"""
        try:
            # Calculate local variance as roughness
            if CV2_AVAILABLE:
                kernel = np.ones((3, 3)) / 9
                mean = cv2.filter2D(height_map, -1, kernel)
                variance = cv2.filter2D(height_map**2, -1, kernel) - mean**2
            else:
                # Fallback without OpenCV
                from scipy.ndimage import uniform_filter

                mean = uniform_filter(height_map, size=3)
                variance = uniform_filter(height_map**2, size=3) - mean**2

            # Normalize to 0-1 range
            roughness_map = (variance - variance.min()) / (
                variance.max() - variance.min() + 1e-8
            )

            return roughness_map

        except Exception as e:
            self.logger.error(f"Error generating roughness map: {e}")
            return np.zeros_like(height_map)

    def generate_ao_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generate ambient occlusion map from height map"""
        try:
            # Simple AO calculation using height differences
            ao_map = np.zeros_like(height_map)

            for i in range(1, height_map.shape[0] - 1):
                for j in range(1, height_map.shape[1] - 1):
                    center_height = height_map[i, j]
                    neighbor_heights = height_map[i - 1 : i + 2, j - 1 : j + 2]

                    # Calculate occlusion based on height differences
                    occlusion = np.sum(neighbor_heights > center_height) / 8.0
                    ao_map[i, j] = 1.0 - occlusion

            return ao_map

        except Exception as e:
            self.logger.error(f"Error generating AO map: {e}")
            return np.ones_like(height_map)

    def _generate_mountain_height_map(self) -> np.ndarray:
        """Generate mountain terrain height map"""
        # Create multiple noise layers for mountain terrain
        height_map = np.zeros((self.config.resolution, self.config.resolution))

        # Base mountain shape
        x = np.linspace(-2, 2, self.config.resolution)
        y = np.linspace(-2, 2, self.config.resolution)
        X, Y = np.meshgrid(x, y)

        # Mountain peaks
        height_map += np.exp(-(X**2 + Y**2)) * 50

        # Add noise layers
        for i in range(3):
            scale = 2**i
            noise = np.random.normal(
                0, 1, (self.config.resolution, self.config.resolution)
            )
            height_map += (
                cv2.resize(noise, (self.config.resolution, self.config.resolution))
                * scale
                * 5
            )

        # Smooth the result
        height_map = gaussian_filter(height_map, sigma=2.0)

        return height_map

    def _generate_hills_height_map(self) -> np.ndarray:
        """Generate hills terrain height map"""
        height_map = np.zeros((self.config.resolution, self.config.resolution))

        # Create multiple hill centers
        num_hills = 5
        for _ in range(num_hills):
            center_x = np.random.uniform(0.2, 0.8) * self.config.resolution
            center_y = np.random.uniform(0.2, 0.8) * self.config.resolution
            radius = np.random.uniform(20, 50)
            height = np.random.uniform(10, 30)

            y, x = np.ogrid[: self.config.resolution, : self.config.resolution]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            hill = height * np.exp(-(dist**2) / (2 * radius**2))
            height_map += hill

        return height_map

    def _generate_valley_height_map(self) -> np.ndarray:
        """Generate valley terrain height map"""
        height_map = np.zeros((self.config.resolution, self.config.resolution))

        # Create valley shape
        x = np.linspace(-2, 2, self.config.resolution)
        y = np.linspace(-2, 2, self.config.resolution)
        X, Y = np.meshgrid(x, y)

        # Valley depression
        height_map -= np.exp(-(X**2 + (Y**2) * 0.5)) * 30

        # Add some hills around the valley
        height_map += np.exp(-((X - 1) ** 2 + (Y - 1) ** 2)) * 15
        height_map += np.exp(-((X + 1) ** 2 + (Y + 1) ** 2)) * 15

        return height_map

    def _generate_plateau_height_map(self) -> np.ndarray:
        """Generate plateau terrain height map"""
        height_map = np.ones((self.config.resolution, self.config.resolution)) * 20

        # Add some variation
        noise = np.random.normal(0, 2, (self.config.resolution, self.config.resolution))
        height_map += noise

        # Smooth edges
        height_map = gaussian_filter(height_map, sigma=3.0)

        return height_map

    def _generate_desert_height_map(self) -> np.ndarray:
        """Generate desert terrain height map"""
        height_map = np.zeros((self.config.resolution, self.config.resolution))

        # Add sand dunes
        for i in range(10):
            center_x = np.random.uniform(0, 1) * self.config.resolution
            center_y = np.random.uniform(0, 1) * self.config.resolution
            radius = np.random.uniform(30, 80)
            height = np.random.uniform(5, 15)

            y, x = np.ogrid[: self.config.resolution, : self.config.resolution]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            dune = height * np.exp(-(dist**2) / (2 * radius**2))
            height_map += dune

        return height_map

    def _generate_ocean_height_map(self) -> np.ndarray:
        """Generate ocean terrain height map"""
        height_map = np.zeros((self.config.resolution, self.config.resolution))

        # Add wave patterns
        x = np.linspace(0, 4 * np.pi, self.config.resolution)
        y = np.linspace(0, 4 * np.pi, self.config.resolution)
        X, Y = np.meshgrid(x, y)

        # Ocean waves
        height_map += np.sin(X) * np.cos(Y) * 2
        height_map += np.sin(X * 2) * np.cos(Y * 2) * 1

        return height_map

    def _generate_default_height_map(self) -> np.ndarray:
        """Generate default terrain height map"""
        return np.random.normal(0, 5, (self.config.resolution, self.config.resolution))

    def generate(self, **kwargs) -> Any:
        """Generate terrain data - implemented by subclasses"""
        return self.generate_height_map()
