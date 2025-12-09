#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Test Script for Astronomical Objects

import bpy
import gin

from infinigen.assets.objects.astronomy import (
    AsteroidBeltFactory,
    CometFactory,
    GasGiantFactory,
    IcePlanetFactory,
    MeteorFactory,
    RockyPlanetFactory,
)


def test_astronomical_objects():
    """Test the astronomical objects generation"""

    print("🚀 Testing Astronomical Objects...")

    # Clear existing objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Test Rocky Planet
    print("Creating Rocky Planet...")
    rocky_planet = RockyPlanetFactory(
        factory_seed=1, radius=1.0, resolution=32
    ).create_asset()
    rocky_planet.name = "Test_Rocky_Planet"
    rocky_planet.location = (0, 0, 0)

    # Test Gas Giant
    print("Creating Gas Giant...")
    gas_giant = GasGiantFactory(
        factory_seed=2, radius=1.5, resolution=32
    ).create_asset()
    gas_giant.name = "Test_Gas_Giant"
    gas_giant.location = (5, 0, 0)

    # Test Ice Planet
    print("Creating Ice Planet...")
    ice_planet = IcePlanetFactory(
        factory_seed=3, radius=0.8, resolution=32
    ).create_asset()
    ice_planet.name = "Test_Ice_Planet"
    ice_planet.location = (10, 0, 0)

    # Test Asteroid Belt
    print("Creating Asteroid Belt...")
    asteroid_belt = AsteroidBeltFactory(
        factory_seed=4, asteroid_count=50, radius=15.0
    ).create_asset()
    asteroid_belt.name = "Test_Asteroid_Belt"
    asteroid_belt.location = (20, 0, 0)

    # Test Comet
    print("Creating Comet...")
    comet = CometFactory(factory_seed=5, tail_length=8.0).create_asset()
    comet.name = "Test_Comet"
    comet.location = (25, 0, 0)

    # Test Meteor
    print("Creating Meteor...")
    meteor = MeteorFactory(factory_seed=6, meteor_size=0.2).create_asset()
    meteor.name = "Test_Meteor"
    meteor.location = (30, 0, 0)

    print("✅ Astronomical Objects Test Complete!")
    print(f"Created {len(bpy.context.scene.objects)} objects")

    # List all objects
    for obj in bpy.context.scene.objects:
        print(f"  - {obj.name} at {obj.location}")


if __name__ == "__main__":
    test_astronomical_objects()
