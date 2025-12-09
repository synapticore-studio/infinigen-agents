#!/usr/bin/env python3
"""
Test script for Blender 4.5.3 Sample Operations functionality
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


def test_sample_operations():
    """Test sample operations with different terrain types"""

    print("🧪 Testing Blender 4.5.3 Sample Operations...")

    # Test configurations
    test_configs = [
        TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=256,
            seed=42,
            enable_geometry_baking=False,  # Disable baking for sample operations
        ),
        TerrainConfig(
            terrain_type=TerrainType.HILLS,
            resolution=128,
            seed=123,
            enable_geometry_baking=False,
        ),
        TerrainConfig(
            terrain_type=TerrainType.VALLEY,
            resolution=64,
            seed=456,
            enable_geometry_baking=False,
        ),
    ]

    results = []

    for i, config in enumerate(test_configs):
        print(
            f"\n📍 Testing {config.terrain_type.value} terrain sample operations (config {i+1})..."
        )

        try:
            # Create terrain engine
            engine = ModernTerrainEngine(config)

            # Generate terrain
            result = engine.generate_terrain()

            if result["success"]:
                print(f"✅ {config.terrain_type.value} terrain generated successfully")

                # Check for sample operations modifiers
                terrain_obj = result["terrain_object"]
                if terrain_obj:
                    modifiers = terrain_obj.modifiers
                    sample_modifiers = [
                        mod
                        for mod in modifiers
                        if "TerrainDetail" in mod.name or "TerrainErosion" in mod.name
                    ]

                    print(f"   🔧 Sample Operations Applied:")
                    print(f"      - Total modifiers: {len(modifiers)}")
                    print(
                        f"      - Sample operation modifiers: {len(sample_modifiers)}"
                    )

                    for mod in sample_modifiers:
                        print(f"         - {mod.name}: {mod.type}")

                    # Check mesh detail
                    vertex_count = len(terrain_obj.data.vertices)
                    face_count = len(terrain_obj.data.polygons)

                    print(f"   📊 Mesh Detail:")
                    print(f"      - Vertices: {vertex_count}")
                    print(f"      - Faces: {face_count}")
                    print(
                        f"      - Detail level: {'High' if config.resolution >= 512 else 'Medium'}"
                    )

                    results.append(
                        {
                            "terrain_type": config.terrain_type.value,
                            "success": True,
                            "vertex_count": vertex_count,
                            "face_count": face_count,
                            "sample_modifiers": len(sample_modifiers),
                        }
                    )
                else:
                    print(f"   ⚠️  No terrain object created")
                    results.append(
                        {
                            "terrain_type": config.terrain_type.value,
                            "success": False,
                            "error": "No terrain object created",
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
    print("\n📊 Sample Operations Summary:")
    print("=" * 50)

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")

    if successful_tests:
        # Analyze sample operations patterns
        avg_vertices = sum(r.get("vertex_count", 0) for r in successful_tests) / len(
            successful_tests
        )
        avg_faces = sum(r.get("face_count", 0) for r in successful_tests) / len(
            successful_tests
        )
        avg_modifiers = sum(
            r.get("sample_modifiers", 0) for r in successful_tests
        ) / len(successful_tests)

        print(f"\n🔍 Sample Operations Analysis:")
        print(f"   - Average vertices: {avg_vertices:.0f}")
        print(f"   - Average faces: {avg_faces:.0f}")
        print(f"   - Average sample modifiers: {avg_modifiers:.1f}")

        # Check for erosion effects
        erosion_terrain_types = [
            r["terrain_type"]
            for r in successful_tests
            if r.get("sample_modifiers", 0) > 0
        ]
        print(f"   - Terrain types with erosion: {', '.join(erosion_terrain_types)}")

    if failed_tests:
        print("\n❌ Failed tests:")
        for test in failed_tests:
            print(f"   - {test['terrain_type']}: {test['error']}")

    return results


def test_sample_node_creation():
    """Test direct sample operations node group creation"""

    print("\n🔧 Testing Sample Operations Node Group Creation...")

    try:
        from infinigen.terrain.terrain_engine import Blender4SampleOperations

        sample_ops = Blender4SampleOperations()

        # Create a test node group
        node_group = sample_ops.create_sample_operations_node_group("TestSampleOps")

        if node_group:
            print("✅ Sample operations node group created successfully")
            print(f"   - Name: {node_group.name}")
            print(f"   - Nodes: {len(node_group.nodes)}")
            print(f"   - Links: {len(node_group.links)}")

            # List node types
            node_types = [node.bl_idname for node in node_group.nodes]
            print(f"   - Node types: {', '.join(set(node_types))}")

            # Check for specific sample operation nodes
            sample_nodes = [
                node
                for node in node_group.nodes
                if "Sample" in node.bl_idname or "Raycast" in node.bl_idname
            ]
            print(f"   - Sample operation nodes: {len(sample_nodes)}")

            return True
        else:
            print("❌ Failed to create sample operations node group")
            return False

    except Exception as e:
        print(f"❌ Error testing sample operations node creation: {e}")
        return False


def test_terrain_detail_enhancement():
    """Test terrain detail enhancement on existing objects"""

    print("\n🔍 Testing Terrain Detail Enhancement...")

    try:
        from infinigen.terrain.terrain_engine import Blender4SampleOperations

        sample_ops = Blender4SampleOperations()

        # Test on all mesh objects in the scene
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

        if not mesh_objects:
            print("⚠️  No mesh objects found in scene")
            return False

        print(f"Found {len(mesh_objects)} mesh objects to enhance")

        for i, obj in enumerate(mesh_objects[:2]):  # Test first 2 objects
            print(f"\n   Enhancing {obj.name}...")

            # Test different detail levels
            for detail_level in ["low", "medium", "high"]:
                print(f"      Testing {detail_level} detail level...")

                # Count vertices before enhancement
                vertices_before = len(obj.data.vertices)

                # Apply detail enhancement
                success = sample_ops.enhance_terrain_detail(obj, detail_level)

                if success:
                    # Count vertices after enhancement
                    vertices_after = len(obj.data.vertices)
                    print(
                        f"         ✅ Enhanced: {vertices_before} -> {vertices_after} vertices"
                    )
                else:
                    print(f"         ❌ Enhancement failed")

        return True

    except Exception as e:
        print(f"❌ Error testing terrain detail enhancement: {e}")
        return False


def test_erosion_effects():
    """Test erosion effects on terrain objects"""

    print("\n🌊 Testing Terrain Erosion Effects...")

    try:
        from infinigen.terrain.terrain_engine import Blender4SampleOperations

        sample_ops = Blender4SampleOperations()

        # Test on all mesh objects in the scene
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

        if not mesh_objects:
            print("⚠️  No mesh objects found in scene")
            return False

        print(f"Found {len(mesh_objects)} mesh objects to test erosion on")

        for i, obj in enumerate(mesh_objects[:2]):  # Test first 2 objects
            print(f"\n   Testing erosion on {obj.name}...")

            # Test different erosion strengths
            for strength in [0.1, 0.3, 0.5]:
                print(f"      Testing erosion strength {strength}...")

                # Apply erosion
                success = sample_ops.add_terrain_erosion(obj, strength)

                if success:
                    print(f"         ✅ Erosion applied with strength {strength}")
                else:
                    print(f"         ❌ Erosion failed with strength {strength}")

        return True

    except Exception as e:
        print(f"❌ Error testing erosion effects: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Blender 4.5.3 Sample Operations Test Suite")
    print("=" * 60)

    # Test sample operations
    results = test_sample_operations()

    # Test node group creation
    test_sample_node_creation()

    # Test detail enhancement
    test_terrain_detail_enhancement()

    # Test erosion effects
    test_erosion_effects()

    print("\n✅ Sample Operations test suite completed!")

    # Test erosion effects
    test_erosion_effects()

    print("\n✅ Sample Operations test suite completed!")
