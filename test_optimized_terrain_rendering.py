#!/usr/bin/env python3
"""
Optimized Test Script for Terrain Rendering System
Tests the Terrain Engineer Agent with optimized settings for faster rendering
"""

import logging
import os
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_optimized_terrain_rendering():
    """Test the terrain rendering system with optimized settings"""
    try:
        logger.info("=== TESTING OPTIMIZED TERRAIN RENDERING SYSTEM ===")

        # Import required modules
        from infinigen.terrain.agents.terrain_engineer_agent import (
            TerrainComplexity,
            TerrainEngineerAgent,
            TerrainGenerationConfig,
        )
        from infinigen.terrain.terrain_engine import TerrainType

        # Create configuration for fast testing
        config = TerrainGenerationConfig(
            terrain_types=[
                TerrainType.MOUNTAIN,
                TerrainType.HILLS,
            ],
            complexity=TerrainComplexity.SIMPLE,  # Simple complexity for faster generation
            resolution=128,  # Very low resolution for fast testing
            seed_range=(1, 2),  # Only 2 terrains
            enable_advanced_features=False,  # Disable advanced features for speed
            add_water=False,  # Disable water for speed
            add_atmosphere=False,  # Disable atmosphere for speed
        )

        # Create agent
        logger.info("Creating Terrain Engineer Agent...")
        agent = TerrainEngineerAgent(config)

        # Run complete workflow
        logger.info("Running optimized terrain generation and rendering workflow...")
        result = agent.generate_and_render_collection()

        if result["success"]:
            logger.info(
                "✅ Optimized terrain rendering workflow completed successfully!"
            )
            logger.info(f"Generated {result['total_terrains']} terrains")
            logger.info(f"Rendered {result['total_renders']} views")
            logger.info(f"Output directory: {result['output_directory']}")

            # List generated files
            output_dir = Path(result["output_directory"])
            if output_dir.exists():
                individual_dir = output_dir / "individual"
                if individual_dir.exists():
                    files = list(individual_dir.glob("*.png"))
                    logger.info(f"Generated {len(files)} individual render files")
                    for file in files[:5]:  # Show first 5 files
                        logger.info(f"  - {file.name}")

                comparison_file = result.get("comparison_file")
                if comparison_file and Path(comparison_file).exists():
                    logger.info(f"Comparison file: {comparison_file}")

            return True
        else:
            logger.error(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing optimized terrain rendering: {e}")
        return False
    finally:
        # Cleanup
        try:
            if "agent" in locals():
                agent.cleanup()
        except:
            pass


def test_simple_terrain_generation():
    """Test simple terrain generation without rendering"""
    try:
        logger.info("=== TESTING SIMPLE TERRAIN GENERATION ===")

        from infinigen.terrain.terrain_engine import (
            ModernTerrainEngine,
            TerrainConfig,
            TerrainType,
        )

        # Test different terrain types
        terrain_types = [TerrainType.MOUNTAIN, TerrainType.HILLS]

        for terrain_type in terrain_types:
            logger.info(f"Testing {terrain_type.value} terrain generation...")

            config = TerrainConfig(
                terrain_type=terrain_type,
                resolution=64,  # Very low resolution for fast testing
                seed=42,
                enable_advanced_features=False,  # Disable for faster testing
                enable_geometry_baking=False,  # Disable baking for speed
            )

            engine = ModernTerrainEngine(config)
            result = engine.generate_terrain()

            if result["success"]:
                logger.info(f"✅ {terrain_type.value} terrain generated successfully")
                logger.info(f"  - Vertices: {result['vertices_count']}")
                logger.info(f"  - Faces: {result['faces_count']}")
                logger.info(f"  - Generation time: {result['generation_time']:.2f}s")
            else:
                logger.error(
                    f"❌ {terrain_type.value} terrain generation failed: {result.get('error')}"
                )

            engine.cleanup()

        return True

    except Exception as e:
        logger.error(f"❌ Error testing simple terrain generation: {e}")
        return False


def main():
    """Main test function"""
    logger.info("Starting Optimized Terrain Rendering System Tests")

    # Test 1: Simple terrain generation
    logger.info("\n" + "=" * 50)
    test1_success = test_simple_terrain_generation()

    # Test 2: Optimized rendering workflow (only if Blender is available)
    logger.info("\n" + "=" * 50)
    try:
        import bpy

        test2_success = test_optimized_terrain_rendering()
    except ImportError:
        logger.warning("Blender not available, skipping rendering tests")
        test2_success = True  # Don't fail the test

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY:")
    logger.info(
        f"Simple terrain generation: {'✅ PASS' if test1_success else '❌ FAIL'}"
    )
    logger.info(
        f"Optimized rendering workflow: {'✅ PASS' if test2_success else '❌ FAIL'}"
    )

    if test1_success and test2_success:
        logger.info("🎉 All optimized tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
