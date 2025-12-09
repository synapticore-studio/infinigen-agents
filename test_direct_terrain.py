#!/usr/bin/env python3
"""
Direct Test of Modern Terrain System
Tests the modernized terrain system directly without complex dependencies
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_direct_terrain():
    """Test the modern terrain system directly"""
    try:
        logger.info("=== TESTING DIRECT MODERN TERRAIN SYSTEM ===")

        # Test if we can import the modern terrain engine
        try:
            from infinigen.terrain.engine import (
                ModernTerrainEngine,
                TerrainConfig,
                TerrainType,
            )

            logger.info("✅ Modern terrain engine imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import modern terrain engine: {e}")
            return False

        # Test if we can import terrain tools
        try:
            from tools.terrain_tools import TerrainTools

            logger.info("✅ Terrain tools imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import terrain tools: {e}")
            return False

        # Test terrain generation
        try:
            terrain_tools = TerrainTools(device="cpu")
            logger.info("✅ Terrain tools initialized successfully")

            # Test coarse terrain generation
            result = terrain_tools.generate_coarse_terrain(
                terrain_type="mountain", seed=42, resolution=64
            )

            if result["success"]:
                logger.info("✅ Coarse terrain generation successful")
                logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
                logger.info(
                    f"   - Generation time: {result.get('generation_time', 0):.2f}s"
                )
            else:
                logger.error(
                    f"❌ Coarse terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Terrain generation test failed: {e}")
            return False

        # Test fine terrain generation
        try:
            result = terrain_tools.generate_fine_terrain(
                terrain_type="hills", seed=42, resolution=128
            )

            if result["success"]:
                logger.info("✅ Fine terrain generation successful")
                logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
                logger.info(
                    f"   - Generation time: {result.get('generation_time', 0):.2f}s"
                )
            else:
                logger.error(
                    f"❌ Fine terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Fine terrain generation test failed: {e}")
            return False

        logger.info("🎉 All direct terrain tests PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ Direct terrain test failed: {e}")
        return False


def main():
    """Run the test"""
    logger.info("🧪 Starting Direct Terrain Test")

    success = test_direct_terrain()

    if success:
        logger.info("🎉 Direct terrain test PASSED!")
        return True
    else:
        logger.error("💥 Direct terrain test FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
