#!/usr/bin/env python3
"""
Test script for Blender 4.5.3 Topology Nodes functionality
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


def test_topology_analysis():
    """Test topology analysis with different terrain types"""

    print("🧪 Testing Blender 4.5.3 Topology Nodes...")

    # Test configurations
    test_configs = [
        TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=128,
            seed=42,
            enable_geometry_baking=False,  # Disable baking for topology analysis
        ),
        TerrainConfig(
            terrain_type=TerrainType.HILLS,
            resolution=64,
            seed=123,
            enable_geometry_baking=False,
        ),
        TerrainConfig(
            terrain_type=TerrainType.CAVE,
            resolution=32,
            seed=456,
            enable_geometry_baking=False,
        ),
    ]

    results = []

    for i, config in enumerate(test_configs):
        print(
            f"\n📍 Testing {config.terrain_type.value} terrain topology (config {i+1})..."
        )

        try:
            # Create terrain engine
            engine = ModernTerrainEngine(config)

            # Generate terrain
            result = engine.generate_terrain()

            if result["success"]:
                print(f"✅ {config.terrain_type.value} terrain generated successfully")

                # Analyze topology
                topology_info = result.get("topology_info", {})

                if topology_info:
                    print(f"   📊 Topology Analysis:")
                    print(
                        f"      - Vertices: {topology_info.get('vertex_count', 'N/A')}"
                    )
                    print(f"      - Edges: {topology_info.get('edge_count', 'N/A')}")
                    print(f"      - Faces: {topology_info.get('face_count', 'N/A')}")
                    print(
                        f"      - Manifold: {'✅ Yes' if topology_info.get('is_manifold', False) else '❌ No'}"
                    )
                    print(
                        f"      - Has Boundary: {'✅ Yes' if topology_info.get('has_boundary', False) else '❌ No'}"
                    )
                    print(f"      - Genus: {topology_info.get('genus', 'N/A')}")

                    results.append(
                        {
                            "terrain_type": config.terrain_type.value,
                            "success": True,
                            "topology_info": topology_info,
                        }
                    )
                else:
                    print(f"   ⚠️  No topology information available")
                    results.append(
                        {
                            "terrain_type": config.terrain_type.value,
                            "success": True,
                            "topology_info": {},
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
    print("\n📊 Topology Analysis Summary:")
    print("=" * 50)

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")

    if successful_tests:
        # Analyze topology patterns
        manifold_count = sum(
            1
            for r in successful_tests
            if r.get("topology_info", {}).get("is_manifold", False)
        )
        boundary_count = sum(
            1
            for r in successful_tests
            if r.get("topology_info", {}).get("has_boundary", False)
        )

        print(f"\n🔍 Topology Patterns:")
        print(f"   - Manifold meshes: {manifold_count}/{len(successful_tests)}")
        print(f"   - Meshes with boundaries: {boundary_count}/{len(successful_tests)}")

        # Calculate average genus
        genus_values = [
            r.get("topology_info", {}).get("genus", 0) for r in successful_tests
        ]
        avg_genus = sum(genus_values) / len(genus_values) if genus_values else 0
        print(f"   - Average genus: {avg_genus:.2f}")

    if failed_tests:
        print("\n❌ Failed tests:")
        for test in failed_tests:
            print(f"   - {test['terrain_type']}: {test['error']}")

    return results


def test_topology_node_creation():
    """Test direct topology node group creation"""

    print("\n🔧 Testing Topology Node Group Creation...")

    try:
        from infinigen.terrain.terrain_engine import Blender4TopologyNodes

        topology_nodes = Blender4TopologyNodes()

        # Create a test node group
        node_group = topology_nodes.create_topology_node_group("TestTopology")

        if node_group:
            print("✅ Topology node group created successfully")
            print(f"   - Name: {node_group.name}")
            print(f"   - Nodes: {len(node_group.nodes)}")
            print(f"   - Links: {len(node_group.links)}")

            # List node types
            node_types = [node.bl_idname for node in node_group.nodes]
            print(f"   - Node types: {', '.join(set(node_types))}")

            return True
        else:
            print("❌ Failed to create topology node group")
            return False

    except Exception as e:
        print(f"❌ Error testing topology node creation: {e}")
        return False


def test_mesh_analysis():
    """Test mesh analysis on existing objects"""

    print("\n🔍 Testing Mesh Analysis on Existing Objects...")

    try:
        from infinigen.terrain.terrain_engine import Blender4TopologyNodes

        topology_nodes = Blender4TopologyNodes()

        # Test on all mesh objects in the scene
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

        if not mesh_objects:
            print("⚠️  No mesh objects found in scene")
            return False

        print(f"Found {len(mesh_objects)} mesh objects to analyze")

        for i, obj in enumerate(mesh_objects[:3]):  # Test first 3 objects
            print(f"\n   Analyzing {obj.name}...")

            topology_info = topology_nodes.analyze_mesh_topology(obj)

            if topology_info:
                print(f"      - Vertices: {topology_info.get('vertex_count', 'N/A')}")
                print(f"      - Edges: {topology_info.get('edge_count', 'N/A')}")
                print(f"      - Faces: {topology_info.get('face_count', 'N/A')}")
                print(f"      - Manifold: {topology_info.get('is_manifold', 'N/A')}")
                print(
                    f"      - Has Boundary: {topology_info.get('has_boundary', 'N/A')}"
                )
                print(f"      - Genus: {topology_info.get('genus', 'N/A')}")
            else:
                print(f"      ❌ Analysis failed")

        return True

    except Exception as e:
        print(f"❌ Error testing mesh analysis: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Blender 4.5.3 Topology Nodes Test Suite")
    print("=" * 60)

    # Test topology analysis
    results = test_topology_analysis()

    # Test node group creation
    test_topology_node_creation()

    # Test mesh analysis
    test_mesh_analysis()

    print("\n✅ Topology Nodes test suite completed!")

    # Test mesh analysis
    test_mesh_analysis()

    print("\n✅ Topology Nodes test suite completed!")
