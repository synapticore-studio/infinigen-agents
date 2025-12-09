#!/usr/bin/env python3
"""
Astronomy Workflow using Infinigen's Core System
Proper integration with Infinigen's export and rendering pipeline
"""

from pathlib import Path
from typing import Any, Dict, List

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


class AstronomyWorkflow:
    """Main workflow class for astronomy scene generation using Infinigen's core system"""

    def __init__(self, output_folder: str = "astronomy_output", scene_seed: int = 42):
        self.output_folder = Path(output_folder)
        self.scene_seed = scene_seed
        self.objects = []

    def clear_scene(self):
        """Clear the current scene"""
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        print("✅ Scene cleared")

    def generate_astronomy_objects(self) -> List[bpy.types.Object]:
        """Generate astronomy objects using Infinigen factories"""

        print("🌌 Generating Astronomy Objects...")

        # 1. Star Field (background)
        print("Creating Star Field...")
        star_field = StarFieldFactory(
            factory_seed=self.scene_seed + 1, star_count=5000, radius=200.0
        ).create_asset()
        star_field.name = "Star_Field"
        star_field.location = (0, 0, 0)
        self.objects.append(star_field)

        # 2. Rocky Planet
        print("Creating Rocky Planet...")
        rocky_planet = RockyPlanetFactory(
            factory_seed=self.scene_seed + 2, radius=1.0, resolution=64
        ).create_asset()
        rocky_planet.name = "Rocky_Planet"
        rocky_planet.location = (0, 0, 0)
        self.objects.append(rocky_planet)

        # 3. Gas Giant with Atmosphere
        print("Creating Gas Giant...")
        gas_giant = GasGiantFactory(
            factory_seed=self.scene_seed + 3,
            radius=2.0,
            resolution=64,
            has_atmosphere=True,
        ).create_asset()
        gas_giant.name = "Gas_Giant"
        gas_giant.location = (15, 0, 0)
        self.objects.append(gas_giant)

        # 4. Ringed Planet
        print("Creating Ringed Planet...")
        ringed_planet = RingedPlanetFactory(
            factory_seed=self.scene_seed + 4, radius=1.8, resolution=64
        ).create_asset()
        ringed_planet.name = "Ringed_Planet"
        ringed_planet.location = (30, 0, 0)
        self.objects.append(ringed_planet)

        # 5. Moon
        print("Creating Moon...")
        moon = MoonFactory(
            factory_seed=self.scene_seed + 5,
            radius=0.3,
            orbit_radius=3.0,
            parent_planet=rocky_planet,
        ).create_asset()
        moon.name = "Moon"
        moon.location = (3, 0, 0)
        self.objects.append(moon)

        # 6. Asteroid Belt
        print("Creating Asteroid Belt...")
        asteroid_belt = AsteroidBeltFactory(
            factory_seed=self.scene_seed + 6, asteroid_count=100, radius=25.0
        ).create_asset()
        asteroid_belt.name = "Asteroid_Belt"
        asteroid_belt.location = (45, 0, 0)
        self.objects.append(asteroid_belt)

        # 7. Comet
        print("Creating Comet...")
        comet = CometFactory(
            factory_seed=self.scene_seed + 7, tail_length=15.0
        ).create_asset()
        comet.name = "Comet"
        comet.location = (60, 0, 0)
        self.objects.append(comet)

        # 8. Space Station
        print("Creating Space Station...")
        space_station = SpaceStationFactory(
            factory_seed=self.scene_seed + 8
        ).create_asset()
        space_station.name = "Space_Station"
        space_station.location = (75, 0, 0)
        self.objects.append(space_station)

        print(f"✅ Generated {len(self.objects)} astronomy objects")
        return self.objects

    def setup_lighting(self):
        """Setup space lighting"""
        print("Setting up space lighting...")

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                obj.select_set(True)
        bpy.ops.object.delete(use_global=False)

        # Add sun light
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
        sun = bpy.context.active_object
        sun.name = "Sun_Light"
        sun.data.energy = 5.0
        sun.data.color = (1.0, 0.95, 0.8)

        # Add ambient light
        bpy.ops.object.light_add(type="AREA", location=(0, 0, 30))
        ambient = bpy.context.active_object
        ambient.name = "Ambient_Light"
        ambient.data.energy = 0.5
        ambient.data.size = 20.0

        print("✅ Space lighting setup complete")

    def setup_camera(self):
        """Setup camera for space scene"""
        print("Setting up camera...")

        # Clear existing cameras
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                obj.select_set(True)
        bpy.ops.object.delete(use_global=False)

        # Add camera
        bpy.ops.object.camera_add(location=(0, -20, 5))
        camera = bpy.context.active_object
        camera.name = "Space_Camera"
        camera.rotation_euler = (1.1, 0, 0)  # Look at the scene

        # Set as active camera
        bpy.context.scene.camera = camera

        print("✅ Camera setup complete")

    def export_scene(self, tasks: List[str] = None):
        """Export scene using Infinigen's core system"""

        if tasks is None:
            tasks = [Task.Export, Task.MeshSave]

        print(f"📤 Exporting scene with tasks: {tasks}")

        # Initialize Infinigen core (not needed for basic export)

        # Create output directory
        self.output_folder.mkdir(exist_ok=True)

        # Triangulate meshes
        print("Triangulating meshes...")
        triangulate_meshes()

        # Save blend file
        blend_file = self.output_folder / "astronomy_scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
        print(f"💾 Scene saved as {blend_file}")

        # Export using Infinigen's system
        if Task.Export in tasks:
            print("Exporting to USD...")
            export_scene(blend_file, self.output_folder)

        if Task.MeshSave in tasks:
            print("Saving meshes...")
            frame_folder = self.output_folder / "frame_0001"
            frame_folder.mkdir(exist_ok=True)

            previous_frame_mesh_id_mapping = dict()
            current_frame_mesh_id_mapping = {}

            exporting.save_obj_and_instances(
                frame_folder / "static_mesh",
                previous_frame_mesh_id_mapping,
                current_frame_mesh_id_mapping,
            )

        print(f"✅ Export complete! Check: {self.output_folder.absolute()}")
        print(f"📁 Output structure:")
        print(f"   - {blend_file} (Blender scene)")
        print(
            f"   - {self.output_folder / 'export_astronomy_scene.blend'} (USD export)"
        )
        print(f"   - {self.output_folder / 'frame_0001/static_mesh'} (NPZ mesh files)")
        print(f"   - {self.output_folder / 'textures'} (Texture files)")

    def render_scene(self, resolution=(1920, 1080), samples=128):
        """Render the astronomy scene"""

        print(f"🎬 Rendering scene at {resolution} with {samples} samples...")

        # Set render settings
        scene = bpy.context.scene
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True

        # Set output path
        output_path = self.output_folder / "astronomy_render.png"
        scene.render.filepath = str(output_path)

        # Render
        bpy.ops.render.render(write_still=True)

        print(f"✅ Render complete! Saved to: {output_path}")
        return output_path

    def run_full_workflow(self):
        """Run the complete astronomy workflow"""

        print("🚀 Starting Complete Astronomy Workflow...")

        # Initialize Infinigen (not needed for basic workflow)

        # Clear and setup scene
        self.clear_scene()
        self.setup_lighting()
        self.setup_camera()

        # Generate objects
        self.generate_astronomy_objects()

        # Export scene
        self.export_scene()

        # Render scene
        self.render_scene()

        print("🎉 Astronomy Workflow Complete!")
        print(f"📁 All outputs saved to: {self.output_folder.absolute()}")


def main():
    """Main function"""

    # Create workflow instance
    workflow = AstronomyWorkflow(
        output_folder="astronomy_complete_output", scene_seed=42
    )

    # Run complete workflow
    workflow.run_full_workflow()


if __name__ == "__main__":
    main()
