#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Objects Generation Example for Infinigen

from pathlib import Path

import bpy
import gin
import numpy as np
from numpy.random import uniform

# Note: Lighting will be set up directly in the script
from infinigen.assets.objects.astronomy import (
    AsteroidBeltFactory,
    CometFactory,
    DarkNebulaFactory,
    EmissionNebulaFactory,
    GalaxyFactory,
    GasGiantFactory,
    IcePlanetFactory,
    MeteorFactory,
    MoonFactory,
    PlanetaryNebulaFactory,
    ReflectionNebulaFactory,
    RingedPlanetFactory,
    RockyPlanetFactory,
    SatelliteFactory,
    SpacecraftFactory,
    SpaceStationFactory,
    StarFieldFactory,
    SupernovaRemnantFactory,
)
from infinigen.core.util import exporting
from infinigen.tools.export import triangulate_meshes


def generate_solar_system():
    """Generate a complete solar system with planets, asteroids, and space objects"""

    # Clear existing objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Generate Sun (Star Field)
    sun = StarFieldFactory(factory_seed=1, star_count=5000, radius=5.0).create_asset()
    sun.name = "Sun"
    sun.location = (0, 0, 0)

    # Generate Rocky Planets
    mercury = RockyPlanetFactory(
        factory_seed=2, radius=0.4, resolution=32
    ).create_asset()
    mercury.name = "Mercury"
    mercury.location = (8, 0, 0)

    venus = RockyPlanetFactory(factory_seed=3, radius=0.6, resolution=48).create_asset()
    venus.name = "Venus"
    venus.location = (12, 0, 0)

    earth = RockyPlanetFactory(factory_seed=4, radius=0.6, resolution=64).create_asset()
    earth.name = "Earth"
    earth.location = (16, 0, 0)

    mars = RockyPlanetFactory(factory_seed=5, radius=0.5, resolution=48).create_asset()
    mars.name = "Mars"
    mars.location = (20, 0, 0)

    # Generate Gas Giants
    jupiter = GasGiantFactory(
        factory_seed=6, radius=2.0, resolution=64, has_atmosphere=True
    ).create_asset()
    jupiter.name = "Jupiter"
    jupiter.location = (30, 0, 0)

    saturn = RingedPlanetFactory(
        factory_seed=7, radius=1.8, resolution=64
    ).create_asset()
    saturn.name = "Saturn"
    saturn.location = (40, 0, 0)

    # Generate Ice Planets
    uranus = IcePlanetFactory(factory_seed=8, radius=1.2, resolution=48).create_asset()
    uranus.name = "Uranus"
    uranus.location = (50, 0, 0)

    neptune = IcePlanetFactory(factory_seed=9, radius=1.1, resolution=48).create_asset()
    neptune.name = "Neptune"
    neptune.location = (60, 0, 0)

    # Generate Moons
    earth_moon = MoonFactory(
        factory_seed=100, radius=0.3, orbit_radius=3.0
    ).create_asset()
    earth_moon.name = "Earth_Moon"
    earth_moon.location = (19, 0, 0)  # Earth + 3 units

    jupiter_moon1 = MoonFactory(
        factory_seed=101, radius=0.2, orbit_radius=4.0
    ).create_asset()
    jupiter_moon1.name = "Jupiter_Moon_1"
    jupiter_moon1.location = (34, 0, 0)  # Jupiter + 4 units

    jupiter_moon2 = MoonFactory(
        factory_seed=102, radius=0.15, orbit_radius=5.5
    ).create_asset()
    jupiter_moon2.name = "Jupiter_Moon_2"
    jupiter_moon2.location = (35.5, 0, 0)  # Jupiter + 5.5 units

    saturn_moon1 = MoonFactory(
        factory_seed=103, radius=0.25, orbit_radius=3.5
    ).create_asset()
    saturn_moon1.name = "Saturn_Moon_1"
    saturn_moon1.location = (43.5, 0, 0)  # Saturn + 3.5 units

    saturn_moon2 = MoonFactory(
        factory_seed=104, radius=0.18, orbit_radius=4.8
    ).create_asset()
    saturn_moon2.name = "Saturn_Moon_2"
    saturn_moon2.location = (44.8, 0, 0)  # Saturn + 4.8 units

    # Generate Asteroid Belt
    asteroid_belt = AsteroidBeltFactory(
        factory_seed=10, asteroid_count=200, radius=25.0
    ).create_asset()
    asteroid_belt.name = "Asteroid Belt"
    asteroid_belt.location = (25, 0, 0)

    # Generate Space Objects
    space_station = SpaceStationFactory(
        factory_seed=11, station_type="modular", size=3.0
    ).create_asset()
    space_station.name = "Space Station"
    space_station.location = (18, 5, 0)

    satellite = SatelliteFactory(
        factory_seed=12, satellite_type="communication", size=0.5
    ).create_asset()
    satellite.name = "Satellite"
    satellite.location = (16, 3, 0)

    spacecraft = SpacecraftFactory(
        factory_seed=13, spacecraft_type="shuttle", size=2.0
    ).create_asset()
    spacecraft.name = "Spacecraft"
    spacecraft.location = (20, 8, 0)

    print("✅ Solar System generated successfully!")
    print(f"Generated {len(bpy.context.scene.objects)} astronomical objects")


def generate_deep_space():
    """Generate deep space objects like nebulae and galaxies"""

    # Generate Nebulae
    emission_nebula = EmissionNebulaFactory(
        factory_seed=20, nebula_size=80.0, density=0.6
    ).create_asset()
    emission_nebula.name = "Emission Nebula"
    emission_nebula.location = (100, 0, 0)

    reflection_nebula = ReflectionNebulaFactory(
        factory_seed=21, nebula_size=60.0, dust_density=0.4
    ).create_asset()
    reflection_nebula.name = "Reflection Nebula"
    reflection_nebula.location = (150, 0, 0)

    dark_nebula = DarkNebulaFactory(
        factory_seed=22, nebula_size=70.0, opacity=0.7
    ).create_asset()
    dark_nebula.name = "Dark Nebula"
    dark_nebula.location = (200, 0, 0)

    planetary_nebula = PlanetaryNebulaFactory(
        factory_seed=23, nebula_size=40.0, shell_thickness=0.4
    ).create_asset()
    planetary_nebula.name = "Planetary Nebula"
    planetary_nebula.location = (250, 0, 0)

    supernova_remnant = SupernovaRemnantFactory(
        factory_seed=24, remnant_size=120.0, shock_strength=0.8
    ).create_asset()
    supernova_remnant.name = "Supernova Remnant"
    supernova_remnant.location = (300, 0, 0)

    # Generate Galaxy
    galaxy = GalaxyFactory(
        factory_seed=25, galaxy_type="spiral", radius=150.0
    ).create_asset()
    galaxy.name = "Galaxy"
    galaxy.location = (400, 0, 0)

    print("✅ Deep Space objects generated successfully!")
    print(f"Generated {len(bpy.context.scene.objects)} deep space objects")


def generate_asteroid_field():
    """Generate an asteroid field with comets and meteors"""

    # Generate Asteroid Field
    asteroid_field = AsteroidBeltFactory(
        factory_seed=30, asteroid_count=500, radius=50.0
    ).create_asset()
    asteroid_field.name = "Asteroid Field"
    asteroid_field.location = (0, 50, 0)

    # Generate Comets
    comets = []
    for i in range(5):
        comet = CometFactory(
            factory_seed=31 + i, tail_length=uniform(5, 15)
        ).create_asset()
        comet.name = f"Comet_{i+1}"
        comet.location = (uniform(-100, 100), uniform(-100, 100), uniform(-50, 50))
        comets.append(comet)

    # Generate Meteors
    meteors = []
    for i in range(20):
        meteor = MeteorFactory(
            factory_seed=40 + i, meteor_size=uniform(0.05, 0.2)
        ).create_asset()
        meteor.name = f"Meteor_{i+1}"
        meteor.location = (uniform(-200, 200), uniform(-200, 200), uniform(-100, 100))
        meteors.append(meteor)

    print("✅ Asteroid Field generated successfully!")
    print(f"Generated {len(comets)} comets and {len(meteors)} meteors")


def setup_scene_and_lighting():
    """Setup scene render settings and lighting for astronomical objects"""

    # Configure render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True

    # Set film transparent for space background
    scene.render.film_transparent = True

    # Clear existing lighting
    bpy.ops.object.select_all(action="SELECT")
    for obj in bpy.context.selected_objects:
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add space-appropriate lighting
    print("Setting up space lighting...")

    # Initialize Infinigen core (not needed for basic export)
    print("✅ Infinigen core ready")

    # Add some point lights for better illumination of objects
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
    sun_light = bpy.context.active_object
    sun_light.name = "Space_Sun_Light"
    sun_light.data.energy = 5.0
    sun_light.data.color = (1.0, 0.95, 0.8)  # Warm sunlight

    # Add ambient lighting
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 30))
    ambient_light = bpy.context.active_object
    ambient_light.name = "Space_Ambient_Light"
    ambient_light.data.energy = 2.0
    ambient_light.data.color = (0.8, 0.9, 1.0)  # Cool ambient
    ambient_light.data.size = 20.0


def main():
    """Main function to generate astronomical objects"""

    print("🚀 Starting Astronomical Objects Generation...")

    # Setup scene and lighting first
    setup_scene_and_lighting()

    # Load configuration (optional)
    try:
        gin.parse_config_file("infinigen_examples/configs_astronomy/planets.gin")
        gin.parse_config_file("infinigen_examples/configs_astronomy/space_objects.gin")
        gin.parse_config_file("infinigen_examples/configs_astronomy/nebulae.gin")
        print("✅ Configuration files loaded")
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  Configuration error: {e}")
        print("Continuing with default settings...")

    # Generate different types of astronomical objects
    generate_solar_system()
    generate_deep_space()
    generate_asteroid_field()

    # Export the scene
    output_path = export_astronomy_scene()

    print("🎉 Astronomical Objects Generation Complete!")
    print(f"Total objects in scene: {len(bpy.context.scene.objects)}")
    print(f"📁 All outputs saved to: {output_path.absolute()}")

    # Print render settings
    scene = bpy.context.scene
    print(f"Render engine: {scene.render.engine}")
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"Samples: {scene.cycles.samples}")
    print(f"Film transparent: {scene.render.film_transparent}")


def export_astronomy_scene(output_folder="astronomy_generated_output"):
    """Export the generated astronomy scene using Infinigen's system"""

    print(f"📤 Exporting astronomy scene to {output_folder}...")

    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Triangulate meshes
    print("Triangulating meshes...")
    triangulate_meshes()

    # Save blend file
    blend_file = output_path / "astronomy_generated_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    print(f"💾 Scene saved as {blend_file}")

    # Export meshes
    print("Exporting meshes...")
    frame_folder = output_path / "frame_0001"
    frame_folder.mkdir(exist_ok=True)

    previous_frame_mesh_id_mapping = dict()
    current_frame_mesh_id_mapping = {}

    exporting.save_obj_and_instances(
        frame_folder / "static_mesh",
        previous_frame_mesh_id_mapping,
        current_frame_mesh_id_mapping,
    )

    print(f"✅ Export complete! Check: {output_path.absolute()}")
    return output_path


if __name__ == "__main__":
    main()
    main()
