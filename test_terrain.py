#!/usr/bin/env python3
"""
Test script for Infinigen Terrain System
Validates that all terrain components are working correctly
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_terrain_libraries():
    """Test if terrain libraries are compiled and accessible"""
    logger.info("Testing terrain libraries...")
    
    terrain_lib_path = Path("infinigen/terrain/lib")
    if not terrain_lib_path.exists():
        logger.error(f"Terrain lib directory not found: {terrain_lib_path}")
        return False
    
    # Check CPU libraries
    cpu_elements = terrain_lib_path / "cpu" / "elements"
    if not cpu_elements.exists():
        logger.error(f"CPU elements directory not found: {cpu_elements}")
        return False
    
    required_libs = [
        "waterbody.so",
        "ground.so", 
        "mountains.so",
        "atmosphere.so"
    ]
    
    missing_libs = []
    for lib in required_libs:
        lib_path = cpu_elements / lib
        if not lib_path.exists():
            missing_libs.append(lib)
            logger.warning(f"Missing library: {lib_path}")
    
    if missing_libs:
        logger.error(f"Missing required libraries: {missing_libs}")
        return False
    
    logger.info("✅ All required terrain libraries found")
    return True

def test_terrain_import():
    """Test if terrain modules can be imported"""
    logger.info("Testing terrain imports...")
    
    try:
        import infinigen.terrain.core
        logger.info("✅ Terrain core imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import terrain core: {e}")
        return False
    
    try:
        from infinigen.terrain.elements import ground, mountains, waterbody
        logger.info("✅ Terrain elements imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import terrain elements: {e}")
        return False
    
    return True

def test_blender_integration():
    """Test if Blender integration works"""
    logger.info("Testing Blender integration...")
    
    try:
        import bpy
        logger.info(f"✅ Blender version: {bpy.app.version_string}")
        
        # Test if we can create a basic scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Create a simple cube to test
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        cube = bpy.context.active_object
        logger.info(f"✅ Created test object: {cube.name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Blender integration failed: {e}")
        return False

def test_gpu_support():
    """Test if GPU support is available"""
    logger.info("Testing GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("⚠️ CUDA not available, using CPU")
            return True
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
        return True

def test_simple_terrain_generation():
    """Test basic terrain generation"""
    logger.info("Testing basic terrain generation...")
    
    try:
        from infinigen.core.util.organization import ElementNames
        from infinigen.terrain.core import Terrain
        
        # Create a simple terrain instance
        terrain = Terrain(
            seed=42,
            task="coarse",
            on_the_fly_asset_folder="",
            device="cpu"
        )
        
        logger.info("✅ Terrain instance created successfully")
        logger.info(f"Terrain elements: {list(terrain.elements.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Terrain generation failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Infinigen Terrain System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Terrain Libraries", test_terrain_libraries),
        ("Terrain Imports", test_terrain_import),
        ("Blender Integration", test_blender_integration),
        ("GPU Support", test_gpu_support),
        ("Simple Terrain Generation", test_simple_terrain_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Terrain system is ready.")
        return 0
    else:
        logger.error("💥 Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 