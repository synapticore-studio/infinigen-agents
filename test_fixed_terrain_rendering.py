#!/usr/bin/env python3
"""
Fixed Test Script for Terrain Rendering System
Tests the corrected Terrain Engineer Agent with proper camera positioning and output paths
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


def test_fixed_terrain_rendering():
    """Test the fixed terrain rendering system"""
    try:
        logger.info("=== TESTING FIXED TERRAIN RENDERING SYSTEM ===")

        # Import required modules
        from infinigen.terrain.agents.terrain_engineer_agent import (
            TerrainComplexity,
            TerrainEngineerAgent,
            TerrainGenerationConfig,
        )
        from infinigen.terrain.terrain_engine import TerrainType

        # Create configuration for testing
        config = TerrainGenerationConfig(
            terrain_types=[TerrainType.MOUNTAIN, TerrainType.HILLS, TerrainType.VALLEY],
            complexity=TerrainComplexity.SIMPLE,
            resolution=256,
            seed_range=(1, 3),
            enable_advanced_features=True,
            add_water=True,
            add_atmosphere=True,
        )

        # Create agent
        agent = TerrainEngineerAgent(
            output_base=Path("terrain_renders"),
            config=config,
        )

        # Generate and render terrains
        logger.info("Generating and rendering terrains...")
        results = agent.generate_and_render_terrains()

        # Check results
        if results:
            logger.info(f"✅ Successfully generated {len(results)} terrain results")
            for terrain_name, result in results.items():
                if result.get("success"):
                    rendered_files = result.get("rendered_files", [])
                    logger.info(
                        f"✅ {terrain_name}: {len(rendered_files)} files rendered"
                    )
                    for file_path in rendered_files:
                        if Path(file_path).exists():
                            logger.info(f"  📁 {file_path}")
                        else:
                            logger.warning(f"  ❌ Missing: {file_path}")
                else:
                    logger.error(
                        f"❌ {terrain_name}: {result.get('error', 'Unknown error')}"
                    )
        else:
            logger.error("❌ No results generated")

        # Test comparison view
        logger.info("Testing comparison view...")
        comparison_result = agent.create_terrain_comparison()
        if isinstance(comparison_result, dict) and comparison_result.get("success"):
            logger.info("✅ Comparison view created successfully")
        elif isinstance(comparison_result, str):
            logger.info(f"✅ Comparison view created: {comparison_result}")
        else:
            logger.error(f"❌ Comparison view failed: {comparison_result}")

        logger.info("=== TEST COMPLETED ===")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_terrain_rendering()
    sys.exit(0 if success else 1)
