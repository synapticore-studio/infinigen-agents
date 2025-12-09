#!/usr/bin/env python3
"""
Generate 3D Astronomy Objects using Infinigen's core export system
Proper integration with Infinigen's workflow and export pipeline
"""

from pathlib import Path

import bpy
import gin

from infinigen.assets.objects.astronomy.asteroids import (
    AsteroidBeltFactory,
    CometFactory,
)
from infinigen.assets.objects.astronomy.moons import MoonFactory
from infinigen.assets.objects.astronomy.nebulae import EmissionNebulaFactory
from infinigen.assets.objects.astronomy.planets import (
    GasGiantFactory,
    IcePlanetFactory,
    RingedPlanetFactory,
    RockyPlanetFactory,
)
from infinigen.assets.objects.astronomy.space_stations import SpaceStationFactory
from infinigen.assets.objects.astronomy.stars import StarFieldFactory
from infinigen.core.util import exporting
from infinigen.core.util.organization import Task
from infinigen.tools.export import export_scene, triangulate_meshes


def clear_scene():
    """Clear the current scene"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    print("✅ Scene cleared")


def generate_3d_objects():
    """Generate various 3D astronomy objects using Infinigen factories"""

    print("🌌 Generating 3D Astronomy Objects...")

    # 1. Rocky Planet
    print("Creating Rocky Planet...")
    rocky_planet = RockyPlanetFactory(
        factory_seed=1, radius=1.0, resolution=64
    ).create_asset()
    rocky_planet.name = "Rocky_Planet"
    rocky_planet.location = (0, 0, 0)

    # 2. Gas Giant with Atmosphere
    print("Creating Gas Giant...")
    gas_giant = GasGiantFactory(
        factory_seed=2, radius=2.0, resolution=64, has_atmosphere=True
    ).create_asset()
    gas_giant.name = "Gas_Giant"
    gas_giant.location = (10, 0, 0)

    # 3. Ringed Planet
    print("Creating Ringed Planet...")
    ringed_planet = RingedPlanetFactory(
        factory_seed=3, radius=1.8, resolution=64
    ).create_asset()
    ringed_planet.name = "Ringed_Planet"
    ringed_planet.location = (20, 0, 0)

    # 4. Moon
    print("Creating Moon...")
    moon = MoonFactory(
        factory_seed=4, radius=0.3, orbit_radius=3.0, parent_planet=rocky_planet
    ).create_asset()
    moon.name = "Moon"
    moon.location = (3, 0, 0)

    # 5. Asteroid Belt
    print("Creating Asteroid Belt...")
    asteroid_belt = AsteroidBeltFactory(
        factory_seed=5, asteroid_count=50, radius=15.0
    ).create_asset()
    asteroid_belt.name = "Asteroid_Belt"
    asteroid_belt.location = (30, 0, 0)

    # 6. Comet
    print("Creating Comet...")
    comet = CometFactory(factory_seed=6, tail_length=10.0).create_asset()
    comet.name = "Comet"
    comet.location = (40, 0, 0)

    # 7. Space Station
    print("Creating Space Station...")
    space_station = SpaceStationFactory(factory_seed=7).create_asset()
    space_station.name = "Space_Station"
    space_station.location = (50, 0, 0)

    # 8. Star Field
    print("Creating Star Field...")
    star_field = StarFieldFactory(
        factory_seed=8, star_count=1000, radius=100.0
    ).create_asset()
    star_field.name = "Star_Field"
    star_field.location = (0, 0, 0)

    print("✅ 3D Objects created successfully!")
    return [
        rocky_planet,
        gas_giant,
        ringed_planet,
        moon,
        asteroid_belt,
        comet,
        space_station,
        star_field,
    ]


def export_using_infinigen_system(output_folder="astronomy_3d_exports"):
    """Export using Infinigen's core export system"""

    print(f"📤 Exporting using Infinigen's core system to {output_folder}...")

    # Initialize Infinigen core (not needed for basic export)

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Triangulate meshes (required by Infinigen)
    print("Triangulating meshes...")
    triangulate_meshes()

    # Save blend file first
    blend_file = output_path / "astronomy_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    print(f"💾 Blender scene saved as {blend_file}")

    # Use Infinigen's export_scene function for USD export
    print("Exporting scene to USD...")
    export_scene(blend_file, output_path)

    # Use Infinigen's save_obj_and_instances for mesh export
    print("Saving meshes using Infinigen's mesh export...")
    frame_folder = output_path / "frame_0001"
    frame_folder.mkdir(exist_ok=True)

    # Save static meshes
    previous_frame_mesh_id_mapping = dict()
    current_frame_mesh_id_mapping = {}

    exporting.save_obj_and_instances(
        frame_folder / "static_mesh",
        previous_frame_mesh_id_mapping,
        current_frame_mesh_id_mapping,
    )

    print(f"✅ Objects exported to {output_path.absolute()}")
    print(f"📁 Check the following directories:")
    print(f"   - {frame_folder / 'static_mesh'} (NPZ mesh files)")
    print(f"   - {output_path / 'export_astronomy_scene.blend'} (USD export)")
    print(f"   - {blend_file} (Blender scene)")
    print(f"   - {output_path / 'textures'} (Texture files)")


def main():
    """Main function using Infinigen's core export system"""

    print("🚀 Starting 3D Astronomy Object Generation with Infinigen Core...")

    # Clear scene
    clear_scene()

    # Generate objects
    objects = generate_3d_objects()

    # Export using Infinigen's system
    export_using_infinigen_system()

    print("🎉 3D Object Generation Complete!")
    print("📁 All files exported using Infinigen's core export system")


if __name__ == "__main__":
    main()
