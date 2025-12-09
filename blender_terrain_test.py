#!/usr/bin/env python3
"""
Blender Terrain Test
Test script to run in Blender to generate modern terrain
"""

import logging
import sys
from pathlib import Path

import bpy

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_terrain_in_blender():
    """Test terrain generation in Blender"""
    try:
        logger.info("🚀 Testing Modern Terrain in Blender")

        # Clear existing mesh objects
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

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

        # Create terrain configuration
        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=128,
            seed=42,
            use_pytorch_geometric=True,
            use_kernels=True,
            use_duckdb_storage=False,  # Disable for testing
        )

        # Initialize terrain engine
        engine = ModernTerrainEngine(config, device="cpu")
        logger.info("✅ Terrain engine initialized")

        # Generate terrain
        logger.info("🏔️ Generating mountain terrain...")
        result = engine.generate_terrain()

        if result["success"]:
            logger.info("✅ Terrain generation successful!")
            logger.info(f"   - Terrain type: {result['metadata']['terrain_type']}")
            logger.info(f"   - Resolution: {result['metadata']['resolution']}")
            logger.info(
                f"   - Generation time: {result['metadata']['generation_time']:.2f}s"
            )
            logger.info(f"   - Vertices: {result['metadata']['terrain_size']}")
            logger.info(f"   - Tech stack: {result['metadata']['tech_stack']}")

            # Check if terrain object was created
            terrain_obj = result.get("terrain_object")
            if terrain_obj:
                logger.info(f"✅ Terrain object created: {terrain_obj.name}")

                # Set up camera for better view
                bpy.ops.object.camera_add(location=(10, -10, 8))
                camera = bpy.context.active_object
                camera.rotation_euler = (1.1, 0, 0.785)

                # Set up lighting
                bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
                sun = bpy.context.active_object
                sun.data.energy = 3

                # Set camera as active
                bpy.context.scene.camera = camera

                logger.info("✅ Camera and lighting setup complete")

                # Render the terrain
                logger.info("📸 Rendering terrain...")
                bpy.context.scene.render.filepath = str(
                    Path("terrain_test_render.png").absolute()
                )
                bpy.ops.render.render(write_still=True)

                logger.info("✅ Terrain rendered successfully!")
                logger.info(
                    f"📁 Render saved to: {Path('terrain_test_render.png').absolute()}"
                )

            else:
                logger.warning("⚠️ No terrain object created")

            return True
        else:
            logger.error(
                f"❌ Terrain generation failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_terrain_in_blender()
    if success:
        logger.info("🎉 Blender terrain test PASSED!")
    else:
        logger.error("💥 Blender terrain test FAILED!")
