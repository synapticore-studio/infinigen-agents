#!/usr/bin/env python3
"""
Test Modern Examples
Tests the modernized system with examples
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_terrain_agent():
    """Test the terrain engineer agent"""
    try:
        logger.info("🧪 Testing Terrain Engineer Agent")

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
        logger.info("✅ Terrain Engineer Agent initialized")

        # Test terrain generation
        output_folder = Path("test_terrain_output")
        scene_seed = 42

        result = terrain_agent.generate_terrain(
            output_folder=output_folder,
            scene_seed=scene_seed,
            file_manager=file_manager,
            logger_tool=logger_tool,
            seed_manager=seed_manager,
            validation_manager=validation_manager,
            terrain_type="mountain",
            detail_level="fine",
        )

        if result["success"]:
            logger.info("✅ Terrain generation successful!")
            logger.info(f"   - Terrain type: {result['terrain_type']}")
            logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
            logger.info(
                f"   - Generation time: {result.get('generation_time', 0):.2f}s"
            )
            return True
        else:
            logger.error(
                f"❌ Terrain generation failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Terrain agent test failed: {e}")
        return False


def test_orchestrator_agent():
    """Test the orchestrator agent"""
    try:
        logger.info("🧪 Testing Orchestrator Agent")

        # Import required modules
        from agents.orchestrator_agent import OrchestratorAgent
        from deps.knowledge_deps import KnowledgeBaseDep
        from tools.ast_udfs import ASTUDFManagerDep

        # Initialize dependencies
        kb = KnowledgeBaseDep()
        ast_manager = ASTUDFManagerDep()

        # Initialize Orchestrator Agent
        orchestrator = OrchestratorAgent(kb, ast_manager)
        logger.info("✅ Orchestrator Agent initialized")

        # Test workflow execution
        workflow_state = {
            "scene_type": "outdoor",
            "scene_seed": 42,
            "complexity": "medium",
            "quality": "high",
            "parameters": {
                "terrain_type": "mountain",
                "detail_level": "fine",
                "enable_advanced_features": True,
            },
        }

        logger.info("🎬 Testing workflow execution...")
        result = orchestrator.execute_workflow(workflow_state)

        if result["success"]:
            logger.info("✅ Workflow execution successful!")
            logger.info(f"   - Workflow type: {result.get('workflow_type', 'Unknown')}")
            logger.info(f"   - Execution time: {result.get('execution_time', 0):.2f}s")
            return True
        else:
            logger.error(
                f"❌ Workflow execution failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Orchestrator agent test failed: {e}")
        return False


def test_terrain_tools():
    """Test the terrain tools directly"""
    try:
        logger.info("🧪 Testing Terrain Tools")

        # Import required modules
        from tools.terrain_tools import TerrainTools

        # Initialize terrain tools
        terrain_tools = TerrainTools(device="cpu")
        logger.info("✅ Terrain Tools initialized")

        # Test coarse terrain generation
        logger.info("🏔️ Testing coarse terrain generation...")
        result = terrain_tools.generate_coarse_terrain(
            terrain_type="mountain", seed=42, resolution=64
        )

        if result["success"]:
            logger.info("✅ Coarse terrain generation successful!")
            logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
            logger.info(
                f"   - Generation time: {result.get('generation_time', 0):.2f}s"
            )
        else:
            logger.error(
                f"❌ Coarse terrain generation failed: {result.get('error', 'Unknown error')}"
            )
            return False

        # Test fine terrain generation
        logger.info("🏔️ Testing fine terrain generation...")
        result = terrain_tools.generate_fine_terrain(
            terrain_type="hills", seed=42, resolution=128
        )

        if result["success"]:
            logger.info("✅ Fine terrain generation successful!")
            logger.info(f"   - Vertices: {result.get('vertices_count', 0)}")
            logger.info(
                f"   - Generation time: {result.get('generation_time', 0):.2f}s"
            )
            return True
        else:
            logger.error(
                f"❌ Fine terrain generation failed: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Terrain tools test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 Starting Modern Examples Test")

    tests = [
        ("Terrain Tools", test_terrain_tools),
        ("Terrain Agent", test_terrain_agent),
        ("Orchestrator Agent", test_orchestrator_agent),
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
        logger.info("🎉 All modern examples tests PASSED!")
        return True
    else:
        logger.error("💥 Some modern examples tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
