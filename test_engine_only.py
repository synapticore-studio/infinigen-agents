#!/usr/bin/env python3
"""
Test Engine Only
Direct test of the modular terrain engine without agents
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_terrain_engine_direct():
    """Test terrain engine directly"""
    try:
        logger.info("Testing terrain engine directly...")

        from infinigen.terrain.engine import (
            ModernTerrainEngine,
            TerrainConfig,
            TerrainType,
        )

        # Create configuration
        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=64,  # Small for testing
            seed=42,
            use_pytorch_geometric=False,  # Disable for testing
            use_kernels=False,  # Disable for testing
            use_duckdb_storage=False,  # Disable for testing
        )

        # Initialize engine
        engine = ModernTerrainEngine(config)
        logger.info("✅ Engine initialized")

        # Generate terrain
        result = engine.generate_terrain()

        if result["success"]:
            logger.info("✅ Terrain generation successful!")
            logger.info(f"   - Terrain type: {result['metadata']['terrain_type']}")
            logger.info(f"   - Resolution: {result['metadata']['resolution']}")
            logger.info(
                f"   - Generation time: {result['metadata']['generation_time']:.2f}s"
            )
            logger.info(f"   - Tech stack: {result['metadata']['tech_stack']}")

            # Check if terrain object exists
            if result["terrain_object"]:
                logger.info(f"   - Terrain object: {result['terrain_object'].name}")

            # Check maps
            if result["height_map"] is not None:
                logger.info(f"   - Height map shape: {result['height_map'].shape}")

            return True
        else:
            logger.error(f"❌ Terrain generation failed: {result['error']}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            engine.cleanup()
        except:
            pass


def test_different_terrain_types():
    """Test different terrain types"""
    try:
        logger.info("Testing different terrain types...")

        from infinigen.terrain.engine import (
            ModernTerrainEngine,
            TerrainConfig,
            TerrainType,
        )

        terrain_types = [
            TerrainType.MOUNTAIN,
            TerrainType.HILLS,
            TerrainType.VALLEY,
            TerrainType.PLATEAU,
        ]

        results = []

        for terrain_type in terrain_types:
            logger.info(f"Testing {terrain_type.value} terrain...")

            config = TerrainConfig(
                terrain_type=terrain_type,
                resolution=32,  # Very small for fast testing
                seed=42,
                use_pytorch_geometric=False,
                use_kernels=False,
                use_duckdb_storage=False,
            )

            engine = ModernTerrainEngine(config)
            result = engine.generate_terrain()

            if result["success"]:
                logger.info(f"✅ {terrain_type.value} terrain generated successfully")
                results.append(True)
            else:
                logger.error(
                    f"❌ {terrain_type.value} terrain failed: {result['error']}"
                )
                results.append(False)

            engine.cleanup()

        success_rate = sum(results) / len(results)
        logger.info(f"Terrain types success rate: {success_rate:.2%}")

        return success_rate > 0.5  # At least 50% success rate

    except Exception as e:
        logger.error(f"❌ Terrain types test failed: {e}")
        return False


def main():
    """Run tests"""
    logger.info("🧪 Starting Terrain Engine Tests")

    tests = [
        ("Direct Engine Test", test_terrain_engine_direct),
        ("Different Terrain Types", test_different_terrain_types),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*40}")

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
    logger.info(f"\n{'='*40}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*40}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        logger.info("🎉 All tests PASSED!")
        return True
    else:
        logger.error("💥 Some tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
