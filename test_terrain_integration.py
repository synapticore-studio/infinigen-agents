#!/usr/bin/env python3
"""
Test Terrain Integration with Orchestrator Agent
Tests the new modular terrain system with outdoor scene examples
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.terrain_engineer import TerrainEngineerAgent
from deps.core_deps import SeedManagerDep, ValidationManagerDep
from deps.model_deps import ModelProviderDep
from infinigen.terrain.engine import ModernTerrainEngine, TerrainConfig, TerrainType
from tools.file_tools import FileManagerDep, LoggerDep

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_terrain_engineer_integration():
    """Test Terrain Engineer Agent with new modular system"""
    try:
        logger.info("🚀 Testing Terrain Engineer Agent Integration")

        # Initialize dependencies
        seed_manager = SeedManagerDep()
        validation_manager = ValidationManagerDep()
        model_provider = ModelProviderDep()
        file_manager = FileManagerDep()
        logger_tool = LoggerDep()

        # Initialize Terrain Engineer Agent
        terrain_agent = TerrainEngineerAgent(model_provider=model_provider)

        # Test terrain generation
        output_folder = Path("test_terrain_output")
        scene_seed = 42

        logger.info("Generating terrain with Terrain Engineer Agent...")
        result = terrain_agent.generate_terrain(
            output_folder=output_folder,
            scene_seed=scene_seed,
            file_manager=file_manager,
            logger_tool=logger_tool,
            seed_manager=seed_manager,
            validation_manager=validation_manager,
            terrain_type="mountain",
            detail_level="medium",
        )

        if result["success"]:
            logger.info("✅ Terrain Engineer Agent test PASSED!")
            logger.info(f"   - Terrain type: {result['terrain_type']}")
            logger.info(f"   - Detail level: {result['detail_level']}")
            logger.info(f"   - Output folder: {result['output_folder']}")
            return True
        else:
            logger.error(f"❌ Terrain Engineer Agent test FAILED: {result['error']}")
            return False

    except Exception as e:
        logger.error(f"❌ Terrain Engineer Agent test failed: {e}")
        return False


def test_modern_terrain_engine():
    """Test Modern Terrain Engine directly"""
    try:
        logger.info("🚀 Testing Modern Terrain Engine")

        # Create configuration for mountain terrain
        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=128,  # Smaller for testing
            seed=42,
            use_pytorch_geometric=True,
            use_kernels=True,
            use_duckdb_storage=True,
        )

        # Initialize engine
        engine = ModernTerrainEngine(config)
        logger.info("✅ Engine initialized")

        # Generate terrain
        result = engine.generate_terrain()

        if result["success"]:
            logger.info("✅ Modern Terrain Engine test PASSED!")
            logger.info(f"   - Terrain type: {result['metadata']['terrain_type']}")
            logger.info(f"   - Resolution: {result['metadata']['resolution']}")
            logger.info(
                f"   - Generation time: {result['metadata']['generation_time']:.2f}s"
            )
            logger.info(f"   - Tech stack: {result['metadata']['tech_stack']}")

            # Check if terrain object exists
            if result["terrain_object"]:
                logger.info(f"   - Terrain object: {result['terrain_object'].name}")

            return True
        else:
            logger.error(f"❌ Modern Terrain Engine test FAILED: {result['error']}")
            return False

    except Exception as e:
        logger.error(f"❌ Modern Terrain Engine test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            engine.cleanup()
        except:
            pass


def test_outdoor_scene_examples():
    """Test with outdoor scene examples"""
    try:
        logger.info("🚀 Testing Outdoor Scene Examples")

        # Test different terrain types from outdoor examples
        terrain_types = [
            TerrainType.MOUNTAIN,  # Forest scenes
            TerrainType.PLATEAU,  # Desert scenes
            TerrainType.HILLS,  # Arctic scenes
            TerrainType.VALLEY,  # Coastal scenes
        ]

        results = []

        for terrain_type in terrain_types:
            logger.info(f"Testing {terrain_type.value} terrain...")

            config = TerrainConfig(
                terrain_type=terrain_type,
                resolution=64,  # Very small for fast testing
                seed=42,
                use_pytorch_geometric=False,  # Disable for faster testing
                use_kernels=False,  # Disable for faster testing
                use_duckdb_storage=False,  # Disable for faster testing
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
        logger.info(f"Outdoor scene examples success rate: {success_rate:.2%}")

        return success_rate > 0.5  # At least 50% success rate

    except Exception as e:
        logger.error(f"❌ Outdoor scene examples test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🧪 Starting Terrain Integration Tests")

    tests = [
        ("Modern Terrain Engine", test_modern_terrain_engine),
        ("Terrain Engineer Agent", test_terrain_engineer_integration),
        ("Outdoor Scene Examples", test_outdoor_scene_examples),
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
        logger.info("🎉 All tests PASSED!")
        return True
    else:
        logger.error("💥 Some tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
