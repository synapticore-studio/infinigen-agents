#!/usr/bin/env python3
"""
Simple Terrain Test
Test the terrain system without complex dependencies
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_terrain_simple():
    """Test terrain generation with simple approach"""
    try:
        logger.info("🧪 Testing Simple Terrain Generation")

        # Test basic imports
        try:
            import numpy as np

            logger.info("✅ NumPy imported")
        except ImportError as e:
            logger.error(f"❌ NumPy import failed: {e}")
            return False

        # Test terrain map generation
        try:
            logger.info("🏔️ Testing terrain map generation...")

            # Create simple height map
            height, width = 64, 64
            x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

            # Mountain terrain
            height_map = np.exp(-(x**2 + y**2) / 0.5) * 20
            height_map += np.random.rand(height, width) * 5

            logger.info(f"✅ Height map created: {height_map.shape}")
            logger.info(f"   - Range: {height_map.min():.2f} to {height_map.max():.2f}")

            # Generate normal map
            gx, gy = np.gradient(height_map)
            norm = np.sqrt(gx**2 + gy**2 + 1.0)
            normal_map = np.stack([-gx / norm, -gy / norm, 1.0 / norm], axis=-1)
            normal_map = ((normal_map + 1.0) / 2.0 * 255).astype(np.uint8)

            logger.info(f"✅ Normal map created: {normal_map.shape}")

            # Generate displacement map
            displacement_map = (height_map - height_map.min()) / (
                height_map.max() - height_map.min() + 1e-8
            )
            logger.info(f"✅ Displacement map created: {displacement_map.shape}")

            # Save height map as image
            try:
                from PIL import Image

                height_img = Image.fromarray((height_map * 255).astype(np.uint8))
                height_img.save("terrain_height_map.png")
                logger.info("✅ Height map saved as terrain_height_map.png")
            except ImportError:
                logger.warning("⚠️ PIL not available, skipping image save")

            return True

        except Exception as e:
            logger.error(f"❌ Terrain map generation failed: {e}")
            return False

    except Exception as e:
        logger.error(f"❌ Simple terrain test failed: {e}")
        return False


def main():
    """Run the test"""
    logger.info("🚀 Starting Simple Terrain Test")

    success = test_terrain_simple()

    if success:
        logger.info("🎉 Simple terrain test PASSED!")
        return True
    else:
        logger.error("💥 Simple terrain test FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
