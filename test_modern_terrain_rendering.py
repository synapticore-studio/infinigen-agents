#!/usr/bin/env python3
"""
Test Modern Terrain Rendering System
Tests the modernized terrain system with proper rendering
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_modern_terrain_rendering():
    """Test the modern terrain rendering system"""
    try:
        logger.info("=== TESTING MODERN TERRAIN RENDERING SYSTEM ===")

        # Import required modules
        from agents.terrain_engineer import TerrainEngineerAgent
        from deps.core_deps import SeedManagerDep, ValidationManagerDep
        from deps.model_deps import ModelProviderDep
        from tools.file_tools import FileManagerDep, LoggerDep

        # Initialize dependencies
        seed_manager = SeedManagerDep()
        validation_manager = ValidationManagerDep()
        model_provider = ModelProviderDep()
        file_manager = FileManagerDep()
        logger_tool = LoggerDep()

        # Initialize Terrain Engineer Agent
        terrain_agent = TerrainEngineerAgent(model_provider=model_provider)

        # Test different terrain types
        terrain_types = ["mountain", "hills", "valley", "plateau"]
        output_folder = Path("terrain_renders_modern")
        scene_seed = 42

        logger.info("Generating modern terrains...")

        results = {}
        for terrain_type in terrain_types:
            logger.info(f"Generating {terrain_type} terrain...")

            result = terrain_agent.generate_terrain(
                output_folder=output_folder / terrain_type,
                scene_seed=scene_seed,
                file_manager=file_manager,
                logger_tool=logger_tool,
                seed_manager=seed_manager,
                validation_manager=validation_manager,
                terrain_type=terrain_type,
                detail_level="fine",
            )

            if result["success"]:
                logger.info(f"✅ {terrain_type} terrain generated successfully")
                logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
                logger.info(
                    f"   - Generation time: {result.get('generation_time', 0):.2f}s"
                )
                results[terrain_type] = result
            else:
                logger.error(
                    f"❌ {terrain_type} terrain generation failed: {result.get('error', 'Unknown error')}"
                )

        # Check results
        if results:
            logger.info(f"✅ Successfully generated {len(results)} terrain results")
            for terrain_name, result in results.items():
                logger.info(
                    f"   - {terrain_name}: {result.get('vertices_count', 0)} vertices"
                )
            return True
        else:
            logger.error("❌ No terrains were generated successfully")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def main():
    """Run the test"""
    logger.info("🧪 Starting Modern Terrain Rendering Test")

    success = test_modern_terrain_rendering()

    if success:
        logger.info("🎉 Modern terrain rendering test PASSED!")
        return True
    else:
        logger.error("💥 Modern terrain rendering test FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
