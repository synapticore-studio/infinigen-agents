#!/usr/bin/env python3
"""
Generate Modern Terrain with High Quality Rendering
Uses the modernized terrain system for realistic terrain generation
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_modern_terrain():
    """Generate modern terrain using the updated system"""
    try:
        logger.info("🚀 Generating Modern Terrain with High Quality Rendering")

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

        # Test different terrain types with modern system
        terrain_types = ["mountain", "hills", "valley", "plateau"]
        output_folder = Path("terrain_renders_modern")
        scene_seed = 42

        logger.info("🏔️ Generating modern terrains with high quality...")

        results = {}
        for i, terrain_type in enumerate(terrain_types):
            logger.info(
                f"🏔️ Generating {terrain_type} terrain ({i+1}/{len(terrain_types)})..."
            )

            # Generate terrain with fine detail
            result = terrain_agent.generate_terrain(
                output_folder=output_folder / terrain_type,
                scene_seed=scene_seed + i,  # Different seed for each terrain
                file_manager=file_manager,
                logger_tool=logger_tool,
                seed_manager=seed_manager,
                validation_manager=validation_manager,
                terrain_type=terrain_type,
                detail_level="fine",  # Use fine detail for better quality
            )

            if result["success"]:
                logger.info(f"✅ {terrain_type} terrain generated successfully")
                logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
                logger.info(
                    f"   - Generation time: {result.get('generation_time', 0):.2f}s"
                )
                logger.info(
                    f"   - Height map shape: {result.get('height_map_shape', 'N/A')}"
                )
                results[terrain_type] = result
            else:
                logger.error(
                    f"❌ {terrain_type} terrain generation failed: {result.get('error', 'Unknown error')}"
                )

        # Summary
        if results:
            logger.info(
                f"🎉 Successfully generated {len(results)} modern terrain results!"
            )
            logger.info("📊 Summary:")
            for terrain_name, result in results.items():
                logger.info(
                    f"   - {terrain_name}: {result.get('vertices_count', 0)} vertices, {result.get('generation_time', 0):.2f}s"
                )

            logger.info(f"📁 Output saved to: {output_folder.absolute()}")
            return True
        else:
            logger.error("❌ No terrains were generated successfully")
            return False

    except Exception as e:
        logger.error(f"❌ Terrain generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the terrain generation"""
    logger.info("🧪 Starting Modern Terrain Generation")

    success = generate_modern_terrain()

    if success:
        logger.info("🎉 Modern terrain generation completed successfully!")
        return True
    else:
        logger.error("💥 Modern terrain generation failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
