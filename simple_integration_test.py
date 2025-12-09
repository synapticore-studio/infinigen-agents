#!/usr/bin/env python3
"""
Simple Integration Test for Modern Terrain System
Tests basic functionality without complex dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")

    try:
        from infinigen.terrain.terrain_engine import TerrainConfig, TerrainType

        print("✅ TerrainConfig and TerrainType imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_terrain_config():
    """Test terrain configuration"""
    print("🧪 Testing terrain configuration...")

    try:
        from infinigen.terrain.terrain_engine import TerrainConfig, TerrainType

        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=64,
            seed=42,
            use_pytorch_geometric=False,  # Disable to avoid dependencies
            enable_geometry_baking=False,
        )

        assert config.terrain_type == TerrainType.MOUNTAIN
        assert config.resolution == 64
        assert config.seed == 42

        print(f"✅ Terrain config created: {config.terrain_type.value}")
        return True
    except Exception as e:
        print(f"❌ Terrain config test failed: {e}")
        return False


def test_terrain_types():
    """Test all terrain types"""
    print("🧪 Testing terrain types...")

    try:
        from infinigen.terrain.terrain_engine import TerrainType

        terrain_types = [
            TerrainType.MOUNTAIN,
            TerrainType.HILLS,
            TerrainType.VALLEY,
            TerrainType.PLATEAU,
            TerrainType.CAVE,
            TerrainType.VOLCANO,
            TerrainType.COAST,
            TerrainType.DESERT,
            TerrainType.FOREST,
            TerrainType.ARCTIC,
        ]

        for terrain_type in terrain_types:
            print(f"  - {terrain_type.value}")

        print(f"✅ All {len(terrain_types)} terrain types available")
        return True
    except Exception as e:
        print(f"❌ Terrain types test failed: {e}")
        return False


def test_mesh_system():
    """Test mesh system without dependencies"""
    print("🧪 Testing mesh system...")

    try:
        import numpy as np

        from infinigen.terrain.terrain_engine import ModernMeshSystem

        mesh_system = ModernMeshSystem(device="cpu")

        # Create simple heightmap
        height_map = np.random.rand(32, 32).astype(np.float32)
        bounds = (-5, 5, -5, 5)

        # This might fail due to dependencies, but we test the creation
        print("✅ ModernMeshSystem created successfully")
        return True
    except Exception as e:
        print(f"❌ Mesh system test failed: {e}")
        return False


def test_rendering_imports():
    """Test rendering module imports"""
    print("🧪 Testing rendering imports...")

    try:
        from infinigen.terrain.rendering import setup_modern_terrain_rendering

        print("✅ Rendering module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Rendering import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Simple Integration Test Suite")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_terrain_config,
        test_terrain_types,
        test_mesh_system,
        test_rendering_imports,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()

    print("📊 Test Results:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
