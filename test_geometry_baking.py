#!/usr/bin/env python3
"""
Test script for Blender 4.5.3 Geometry Node Baking functionality
"""

import sys
from pathlib import Path

import bpy

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from infinigen.terrain.terrain_engine import (
    ModernTerrainEngine,
    TerrainConfig,
    TerrainType,
)


def test_geometry_baking():
    """Test Geometry Node Baking with different terrain types"""

    print("🧪 Testing Blender 4.5.3 Geometry Node Baking...")

    # Test configurations
    test_configs = [
        TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=256,
            seed=42,
            enable_geometry_baking=True,
        ),
        TerrainConfig(
            terrain_type=TerrainType.HILLS,
            resolution=128,
            seed=123,
            enable_geometry_baking=True,
        ),
        TerrainConfig(
            terrain_type=TerrainType.CAVE,
            resolution=64,
            seed=456,
            enable_geometry_baking=True,
        ),
    ]

    results = []

    for i, config in enumerate(test_configs):
        print(f"\n📍 Testing {config.terrain_type.value} terrain (config {i+1})...")

        try:
            # Create terrain engine
            engine = ModernTerrainEngine(config)

            # Generate terrain
            result = engine.generate_terrain()

            if result["success"]:
                print(f"✅ {config.terrain_type.value} terrain generated successfully")
                print(f"   - Vertices: {result['vertices_count']}")
                print(f"   - Faces: {result['faces_count']}")
                print(f"   - Generation time: {result['generation_time']:.2f}s")

                # Check if baking was applied
                terrain_obj = result["terrain_object"]
                if terrain_obj:
                    # Check for baked geometry
                    has_baked_geometry = any(
                        "baked" in modifier.name.lower()
                        for modifier in terrain_obj.modifiers
                    )
                    print(
                        f"   - Geometry baking: {'✅ Applied' if has_baked_geometry else '❌ Not applied'}"
                    )

                results.append(
                    {
                        "terrain_type": config.terrain_type.value,
                        "success": True,
                        "generation_time": result["generation_time"],
                        "vertices": result["vertices_count"],
                        "faces": result["faces_count"],
                    }
                )
            else:
                print(
                    f"❌ {config.terrain_type.value} terrain generation failed: {result.get('error', 'Unknown error')}"
                )
                results.append(
                    {
                        "terrain_type": config.terrain_type.value,
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                    }
                )

            # Cleanup
            engine.cleanup()

        except Exception as e:
            print(f"❌ Error testing {config.terrain_type.value}: {e}")
            results.append(
                {
                    "terrain_type": config.terrain_type.value,
                    "success": False,
                    "error": str(e),
                }
            )

    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")

    if successful_tests:
        avg_time = sum(r["generation_time"] for r in successful_tests) / len(
            successful_tests
        )
        print(f"⏱️  Average generation time: {avg_time:.2f}s")

    if failed_tests:
        print("\n❌ Failed tests:")
        for test in failed_tests:
            print(f"   - {test['terrain_type']}: {test['error']}")

    return results


def test_baking_performance():
    """Test performance difference with and without baking"""

    print("\n🚀 Testing Geometry Baking Performance...")

    config = TerrainConfig(terrain_type=TerrainType.MOUNTAIN, resolution=512, seed=42)

    # Test without baking
    print("Testing WITHOUT geometry baking...")
    config.enable_geometry_baking = False
    engine_no_baking = ModernTerrainEngine(config)
    result_no_baking = engine_no_baking.generate_terrain()
    engine_no_baking.cleanup()

    # Test with baking
    print("Testing WITH geometry baking...")
    config.enable_geometry_baking = True
    engine_with_baking = ModernTerrainEngine(config)
    result_with_baking = engine_with_baking.generate_terrain()
    engine_with_baking.cleanup()

    if result_no_baking["success"] and result_with_baking["success"]:
        time_no_baking = result_no_baking["generation_time"]
        time_with_baking = result_with_baking["generation_time"]

        print(f"\n📈 Performance Comparison:")
        print(f"   Without baking: {time_no_baking:.2f}s")
        print(f"   With baking: {time_with_baking:.2f}s")

        if time_with_baking < time_no_baking:
            improvement = ((time_no_baking - time_with_baking) / time_no_baking) * 100
            print(f"   🎉 Baking improved performance by {improvement:.1f}%")
        else:
            print(f"   ⚠️  Baking did not improve performance in this test")
    else:
        print("❌ Performance test failed - could not generate terrain")


if __name__ == "__main__":
    print("🧪 Blender 4.5.3 Geometry Node Baking Test Suite")
    print("=" * 60)

    # Test basic functionality
    results = test_geometry_baking()

    # Test performance
    test_baking_performance()

    print("\n✅ Test suite completed!")

    # Test performance
    test_baking_performance()

    print("\n✅ Test suite completed!")
