#!/usr/bin/env python3
"""
Test script for Astronomy SimObjects
"""

from pathlib import Path

import bpy
import gin

from infinigen.assets.sim_objects.satellite import SatelliteFactory

# Import sim objects
from infinigen.assets.sim_objects.space_station import SpaceStationFactory
from infinigen.core.sim.sim_factory import spawn_simready


def test_astronomy_sim_objects():
    """Test astronomy sim objects generation"""

    print("🚀 Testing Astronomy SimObjects...")

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Test Space Station
    print("Creating Space Station...")
    space_station = SpaceStationFactory(factory_seed=1).create_asset()
    space_station.name = "Test_Space_Station"
    space_station.location = (0, 0, 0)

    # Test Satellite
    print("Creating Satellite...")
    satellite = SatelliteFactory(factory_seed=2).create_asset()
    satellite.name = "Test_Satellite"
    satellite.location = (5, 0, 0)

    print("✅ Astronomy SimObjects created successfully!")
    print(f"Space Station: {space_station.name}")
    print(f"Satellite: {satellite.name}")

    return space_station, satellite


def test_sim_export():
    """Test sim object export"""

    print("📤 Testing SimObject Export...")

    try:
        # Export space station
        export_path, semantic_mapping = spawn_simready(
            name="space_station",
            exporter="mjcf",
            export_dir=Path("./astronomy_sim_exports"),
            seed=1,
        )

        print(f"✅ Space Station exported to: {export_path}")
        print(f"Semantic mapping: {semantic_mapping}")

        # Export satellite
        export_path, semantic_mapping = spawn_simready(
            name="satellite",
            exporter="urdf",
            export_dir=Path("./astronomy_sim_exports"),
            seed=2,
        )

        print(f"✅ Satellite exported to: {export_path}")
        print(f"Semantic mapping: {semantic_mapping}")

    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    """Main test function"""

    print("🌌 Starting Astronomy SimObjects Test...")

    # Test object creation
    space_station, satellite = test_astronomy_sim_objects()

    # Test sim export
    test_sim_export()

    print("🎉 Astronomy SimObjects Test Complete!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
