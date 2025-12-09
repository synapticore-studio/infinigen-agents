#!/usr/bin/env python3
"""
Astronomy Film Creator for Infinigen
Creates cinematic animations of astronomical objects using Infinigen's core system
"""

import math
from pathlib import Path

import bpy
import gin
import mathutils
from mathutils import Euler, Vector
from numpy.random import uniform

from infinigen.assets.objects.astronomy.asteroids import (
    AsteroidBeltFactory,
    CometFactory,
)
from infinigen.assets.objects.astronomy.moons import MoonFactory
from infinigen.assets.objects.astronomy.nebulae import EmissionNebulaFactory

# Import astronomy factories
from infinigen.assets.objects.astronomy.planets import (
    GasGiantFactory,
    IcePlanetFactory,
    RingedPlanetFactory,
    RockyPlanetFactory,
)
from infinigen.assets.objects.astronomy.space_stations import SpaceStationFactory
from infinigen.assets.objects.astronomy.stars import StarFieldFactory
from infinigen.core.util import exporting
from infinigen.tools.export import triangulate_meshes


def setup_cinematic_scene():
    """Setup scene for cinematic rendering"""
    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Set render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    # Set frame range for animation
    scene.frame_start = 1
    scene.frame_end = 300  # 10 seconds at 30fps
    scene.frame_current = 1

    # Setup lighting
    setup_space_lighting()

    print("✅ Cinematic scene setup complete")


def setup_space_lighting():
    """Setup space lighting for cinematic effect"""
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT":
            obj.select_set(True)
    bpy.ops.object.delete()

    # Add sun light
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))
    sun_light = bpy.context.active_object
    sun_light.name = "Sun_Light"
    sun_light.data.energy = 10.0
    sun_light.data.color = (1.0, 0.95, 0.8)  # Warm sunlight

    # Add area light for fill
    bpy.ops.object.light_add(type="AREA", location=(50, 50, 50))
    area_light = bpy.context.active_object
    area_light.name = "Fill_Light"
    area_light.data.energy = 5.0
    area_light.data.size = 20.0

    # Add rim light
    bpy.ops.object.light_add(type="AREA", location=(-30, -30, 30))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 3.0
    rim_light.data.size = 10.0

    print("✅ Space lighting setup complete")


def create_solar_system_scene():
    """Create a complete solar system for the film"""
    # Clear existing objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Create Sun
    sun = StarFieldFactory(factory_seed=1, star_count=10000, radius=8.0).create_asset()
    sun.name = "Sun"
    sun.location = (0, 0, 0)

    # Create planets with cinematic spacing
    planets = []

    # Mercury
    mercury = RockyPlanetFactory(
        factory_seed=2, radius=0.4, resolution=32
    ).create_asset()
    mercury.name = "Mercury"
    mercury.location = (12, 0, 0)
    planets.append(mercury)

    # Venus
    venus = RockyPlanetFactory(factory_seed=3, radius=0.6, resolution=48).create_asset()
    venus.name = "Venus"
    venus.location = (18, 0, 0)
    planets.append(venus)

    # Earth with Moon
    earth = RockyPlanetFactory(factory_seed=4, radius=0.6, resolution=64).create_asset()
    earth.name = "Earth"
    earth.location = (24, 0, 0)
    planets.append(earth)

    earth_moon = MoonFactory(
        factory_seed=100, radius=0.2, orbit_radius=2.0
    ).create_asset()
    earth_moon.name = "Earth_Moon"
    earth_moon.location = (26, 0, 0)

    # Mars
    mars = RockyPlanetFactory(factory_seed=5, radius=0.5, resolution=48).create_asset()
    mars.name = "Mars"
    mars.location = (32, 0, 0)
    planets.append(mars)

    # Jupiter with atmosphere
    jupiter = GasGiantFactory(
        factory_seed=6, radius=2.5, resolution=64, has_atmosphere=True
    ).create_asset()
    jupiter.name = "Jupiter"
    jupiter.location = (45, 0, 0)
    planets.append(jupiter)

    # Saturn with rings
    saturn = RingedPlanetFactory(
        factory_seed=7, radius=2.0, resolution=64
    ).create_asset()
    saturn.name = "Saturn"
    saturn.location = (60, 0, 0)
    planets.append(saturn)

    # Uranus
    uranus = IcePlanetFactory(factory_seed=8, radius=1.2, resolution=48).create_asset()
    uranus.name = "Uranus"
    uranus.location = (75, 0, 0)
    planets.append(uranus)

    # Neptune
    neptune = IcePlanetFactory(factory_seed=9, radius=1.1, resolution=48).create_asset()
    neptune.name = "Neptune"
    neptune.location = (90, 0, 0)
    planets.append(neptune)

    # Add asteroid belt
    asteroid_belt = AsteroidBeltFactory(
        factory_seed=10, asteroid_count=500, radius=35.0
    ).create_asset()
    asteroid_belt.name = "Asteroid_Belt"
    asteroid_belt.location = (35, 0, 0)

    # Add comets
    comets = []
    for i in range(3):
        comet = CometFactory(factory_seed=20 + i, tail_length=15.0).create_asset()
        comet.name = f"Comet_{i+1}"
        comet.location = (uniform(20, 80), uniform(-20, 20), uniform(-10, 10))
        comets.append(comet)

    print(
        f"✅ Solar system created with {len(planets)} planets and {len(comets)} comets"
    )
    return planets, comets


def setup_camera_animation():
    """Setup cinematic camera movement"""
    # Create camera
    bpy.ops.object.camera_add(location=(0, -50, 20))
    camera = bpy.context.active_object
    camera.name = "Cinematic_Camera"

    # Set as active camera
    bpy.context.scene.camera = camera

    # Point camera at origin
    camera.rotation_euler = (math.radians(60), 0, 0)

    # Create camera animation
    # Start: Wide shot of solar system
    camera.location = (0, -80, 30)
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)

    # Middle: Close-up of Jupiter
    camera.location = (45, -15, 8)
    camera.keyframe_insert(data_path="location", frame=150)
    camera.rotation_euler = (math.radians(45), 0, 0)
    camera.keyframe_insert(data_path="rotation_euler", frame=150)

    # End: Fly-by of Saturn
    camera.location = (60, -20, 12)
    camera.keyframe_insert(data_path="location", frame=300)
    camera.rotation_euler = (math.radians(30), 0, 0)
    camera.keyframe_insert(data_path="rotation_euler", frame=300)

    # Set interpolation to smooth
    for fcurve in camera.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"
            keyframe.handle_left_type = "AUTO"
            keyframe.handle_right_type = "AUTO"

    print("✅ Camera animation setup complete")


def add_cinematic_effects():
    """Add cinematic effects like depth of field and motion blur"""
    camera = bpy.context.scene.camera

    # Enable depth of field
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 50.0
    camera.data.dof.aperture_fstop = 2.8

    # Enable motion blur
    bpy.context.scene.render.use_motion_blur = True
    bpy.context.scene.render.motion_blur_shutter = 0.5

    print("✅ Cinematic effects enabled")


def render_film(output_folder="astronomy_film_output"):
    """Render the complete film using Infinigen's system"""

    # Initialize Infinigen core (not needed for basic export)

    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    scene = bpy.context.scene

    # Set output path
    film_file = output_path / "astronomy_film"
    scene.render.filepath = str(film_file)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"

    print("🎬 Starting film render...")
    print(f"Frames: {scene.frame_start} to {scene.frame_end}")
    print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"Output: {film_file}")

    # Triangulate meshes before rendering
    print("Triangulating meshes...")
    triangulate_meshes()

    # Render animation
    bpy.ops.render.render(animation=True)

    # Save blend file
    blend_file = output_path / "astronomy_film_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))

    # Export scene data
    print("Exporting scene data...")
    frame_folder = output_path / "frame_0001"
    frame_folder.mkdir(exist_ok=True)

    previous_frame_mesh_id_mapping = dict()
    current_frame_mesh_id_mapping = {}

    exporting.save_obj_and_instances(
        frame_folder / "static_mesh",
        previous_frame_mesh_id_mapping,
        current_frame_mesh_id_mapping,
    )

    print("✅ Film render complete!")
    print(f"📁 Film saved to: {output_path.absolute()}")
    print(f"🎬 Video file: {film_file}")
    print(f"💾 Blender scene: {blend_file}")
    print(f"📦 Mesh data: {frame_folder / 'static_mesh'}")

    return output_path


def main():
    """Main function to create astronomy film"""
    print("🚀 Starting Astronomy Film Creation...")

    # Setup cinematic scene
    setup_cinematic_scene()

    # Create solar system
    planets, comets = create_solar_system_scene()

    # Setup camera animation
    setup_camera_animation()

    # Add cinematic effects
    add_cinematic_effects()

    # Render film
    output_path = render_film()

    print("🎉 Astronomy Film Creation Complete!")
    print(f"📁 All outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
