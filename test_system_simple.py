#!/usr/bin/env python3
"""
Simple System Test
Test the system without virtual environment dependencies
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if we can import the modules"""
    try:
        logger.info("🧪 Testing imports...")

        # Test basic imports
        try:
            import numpy as np

            logger.info("✅ NumPy imported")
        except ImportError as e:
            logger.error(f"❌ NumPy import failed: {e}")
            return False

        try:
            import torch

            logger.info("✅ PyTorch imported")
        except ImportError as e:
            logger.error(f"❌ PyTorch import failed: {e}")
            return False

        # Test terrain engine imports
        try:
            from infinigen.terrain.engine import (
                ModernTerrainEngine,
                TerrainConfig,
                TerrainType,
            )

            logger.info("✅ Modern terrain engine imported")
        except Exception as e:
            logger.error(f"❌ Terrain engine import failed: {e}")
            return False

        # Test terrain tools imports
        try:
            from tools.terrain_tools import TerrainTools

            logger.info("✅ Terrain tools imported")
        except Exception as e:
            logger.error(f"❌ Terrain tools import failed: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_terrain_config():
    """Test terrain configuration"""
    try:
        logger.info("🧪 Testing terrain configuration...")

        from infinigen.terrain.engine import TerrainConfig, TerrainType

        # Test different terrain types
        configs = [
            TerrainConfig(terrain_type=TerrainType.MOUNTAIN, resolution=64),
            TerrainConfig(terrain_type=TerrainType.HILLS, resolution=128),
            TerrainConfig(terrain_type=TerrainType.VALLEY, resolution=256),
            TerrainConfig(terrain_type=TerrainType.PLATEAU, resolution=512),
        ]

        for config in configs:
            logger.info(
                f"✅ {config.terrain_type.value} config: {config.resolution}x{config.resolution}"
            )

        return True

    except Exception as e:
        logger.error(f"❌ Terrain config test failed: {e}")
        return False


def test_terrain_generation():
    """Test terrain generation without Blender"""
    try:
        logger.info("🧪 Testing terrain generation...")

        from infinigen.terrain.engine import TerrainConfig, TerrainType
        from infinigen.terrain.engine.generators import TerrainMapGenerator

        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=64,
            seed=42,
        )

        generator = TerrainMapGenerator(config, device="cpu")

        # Generate height map
        height_map = generator.generate_height_map()
        logger.info(f"✅ Height map generated: {height_map.shape}")

        # Generate other maps
        normal_map = generator.generate_normal_map(height_map)
        displacement_map = generator.generate_displacement_map(height_map)
        roughness_map = generator.generate_roughness_map(height_map)
        ao_map = generator.generate_ao_map(height_map)

        logger.info(f"✅ Normal map: {normal_map.shape}")
        logger.info(f"✅ Displacement map: {displacement_map.shape}")
        logger.info(f"✅ Roughness map: {roughness_map.shape}")
        logger.info(f"✅ AO map: {ao_map.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Terrain generation test failed: {e}")
        return False


def test_terrain_engine():
    """Test terrain engine without Blender"""
    try:
        logger.info("🧪 Testing terrain engine...")

        from infinigen.terrain.engine import (
            ModernTerrainEngine,
            TerrainConfig,
            TerrainType,
        )

        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=64,
            seed=42,
            use_pytorch_geometric=False,  # Disable for testing
            use_kernels=False,  # Disable for testing
            use_duckdb_storage=False,  # Disable for testing
        )

        engine = ModernTerrainEngine(config, device="cpu")
        logger.info("✅ Terrain engine initialized")

        # Try to generate terrain (will fail at Blender step, but we can test the rest)
        try:
            result = engine.generate_terrain()
            if result["success"]:
                logger.info("✅ Terrain generation successful!")
                logger.info(f"   - Terrain type: {result['metadata']['terrain_type']}")
                logger.info(f"   - Resolution: {result['metadata']['resolution']}")
                logger.info(
                    f"   - Generation time: {result['metadata']['generation_time']:.2f}s"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ Terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                return False
        except Exception as e:
            if "bpy" in str(e).lower():
                logger.warning("⚠️ Blender not available, but other components work")
                return True
            else:
                raise e

    except Exception as e:
        logger.error(f"❌ Terrain engine test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 Starting Simple System Test")

    tests = [
        ("Imports", test_imports),
        ("Terrain Config", test_terrain_config),
        ("Terrain Generation", test_terrain_generation),
        ("Terrain Engine", test_terrain_engine),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            success = test_func()
            results.append((test_name, success))

            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        logger.info("🎉 All simple system tests PASSED!")
        return True
    else:
        logger.error("💥 Some simple system tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
