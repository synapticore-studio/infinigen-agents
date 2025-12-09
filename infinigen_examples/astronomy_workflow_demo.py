#!/usr/bin/env python3
"""
Astronomy Workflow Demo
Zeigt alle verfügbaren Workflow-Komponenten
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


class AstronomyWorkflowDemo:
    """Demo-Klasse für alle Astronomy Workflow-Komponenten"""

    def __init__(
        self, output_folder: str = "astronomy_demo_output", scene_seed: int = 42
    ):
        self.output_folder = Path(output_folder)
        self.scene_seed = scene_seed
        self.objects = []

    def demo_1_basic_objects(self):
        """Demo 1: Einfache Objekte erstellen"""
        print("\n=== DEMO 1: Basic Objects ===")

        # Clear scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # Einzelne Objekte
        planet = RockyPlanetFactory(
            factory_seed=1, radius=1.0, resolution=32
        ).create_asset()
        planet.name = "Demo_Planet"
        planet.location = (0, 0, 0)

        moon = MoonFactory(
            factory_seed=2, radius=0.3, orbit_radius=2.0, parent_planet=planet
        ).create_asset()
        moon.name = "Demo_Moon"
        moon.location = (2, 0, 0)

        print("✅ Basic objects created")
        return [planet, moon]

    def demo_2_complex_scene(self):
        """Demo 2: Komplexe Szene mit allen Objekttypen"""
        print("\n=== DEMO 2: Complex Scene ===")

        # Clear scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        objects = []

        # Sternenfeld
        star_field = StarFieldFactory(
            factory_seed=10, star_count=1000, radius=50.0
        ).create_asset()
        star_field.name = "Demo_StarField"
        objects.append(star_field)

        # Planeten
        rocky = RockyPlanetFactory(
            factory_seed=11, radius=1.0, resolution=32
        ).create_asset()
        rocky.name = "Demo_Rocky"
        rocky.location = (0, 0, 0)
        objects.append(rocky)

        gas_giant = GasGiantFactory(
            factory_seed=12, radius=2.0, resolution=32, has_atmosphere=True
        ).create_asset()
        gas_giant.name = "Demo_GasGiant"
        gas_giant.location = (10, 0, 0)
        objects.append(gas_giant)

        ringed = RingedPlanetFactory(
            factory_seed=13, radius=1.5, resolution=32
        ).create_asset()
        ringed.name = "Demo_Ringed"
        ringed.location = (20, 0, 0)
        objects.append(ringed)

        # Asteroiden
        asteroid_belt = AsteroidBeltFactory(
            factory_seed=14, asteroid_count=20, radius=15.0
        ).create_asset()
        asteroid_belt.name = "Demo_Asteroids"
        asteroid_belt.location = (30, 0, 0)
        objects.append(asteroid_belt)

        # Komet
        comet = CometFactory(factory_seed=15, tail_length=8.0).create_asset()
        comet.name = "Demo_Comet"
        comet.location = (40, 0, 0)
        objects.append(comet)

        # Raumstation
        space_station = SpaceStationFactory(factory_seed=16).create_asset()
        space_station.name = "Demo_SpaceStation"
        space_station.location = (50, 0, 0)
        objects.append(space_station)

        print(f"✅ Complex scene created with {len(objects)} objects")
        return objects

    def demo_3_lighting_setup(self):
        """Demo 3: Beleuchtung für Space-Szenen"""
        print("\n=== DEMO 3: Lighting Setup ===")

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                obj.select_set(True)
        bpy.ops.object.delete(use_global=False)

        # Sonnenlicht
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
        sun = bpy.context.active_object
        sun.name = "Demo_Sun"
        sun.data.energy = 5.0
        sun.data.color = (1.0, 0.95, 0.8)

        # Ambient-Licht
        bpy.ops.object.light_add(type="AREA", location=(0, 0, 30))
        ambient = bpy.context.active_object
        ambient.name = "Demo_Ambient"
        ambient.data.energy = 0.5
        ambient.data.size = 20.0

        # Rim-Licht
        bpy.ops.object.light_add(type="AREA", location=(-20, -20, 20))
        rim = bpy.context.active_object
        rim.name = "Demo_Rim"
        rim.data.energy = 2.0
        rim.data.size = 10.0

        print("✅ Space lighting setup complete")

    def demo_4_camera_setup(self):
        """Demo 4: Kamera für astronomische Szenen"""
        print("\n=== DEMO 4: Camera Setup ===")

        # Clear existing cameras
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                obj.select_set(True)
        bpy.ops.object.delete(use_global=False)

        # Kamera hinzufügen
        bpy.ops.object.camera_add(location=(0, -15, 8))
        camera = bpy.context.active_object
        camera.name = "Demo_Camera"
        camera.rotation_euler = (1.1, 0, 0)  # Blick auf die Szene

        # Als aktive Kamera setzen
        bpy.context.scene.camera = camera

        print("✅ Camera setup complete")

    def demo_5_export_system(self, objects: List[bpy.types.Object]):
        """Demo 5: Infinigen Export-System"""
        print("\n=== DEMO 5: Export System ===")

        # Output-Verzeichnis erstellen
        self.output_folder.mkdir(exist_ok=True)

        # Meshes triangulieren
        print("Triangulating meshes...")
        triangulate_meshes()

        # Blender-Datei speichern
        blend_file = self.output_folder / "demo_scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
        print(f"💾 Scene saved as {blend_file}")

        # USD-Export
        print("Exporting to USD...")
        export_scene(blend_file, self.output_folder)

        # Mesh-Export
        print("Exporting meshes...")
        frame_folder = self.output_folder / "frame_0001"
        frame_folder.mkdir(exist_ok=True)

        previous_frame_mesh_id_mapping = dict()
        current_frame_mesh_id_mapping = {}

        exporting.save_obj_and_instances(
            frame_folder / "static_mesh",
            previous_frame_mesh_id_mapping,
            current_frame_mesh_id_mapping,
        )

        print("✅ Export complete!")

    def demo_6_rendering(self):
        """Demo 6: Rendering mit Cycles"""
        print("\n=== DEMO 6: Rendering ===")

        # Render-Einstellungen
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True

        # Output-Pfad
        output_path = self.output_folder / "demo_render.png"
        scene.render.filepath = str(output_path)

        # Rendern
        print("Rendering scene...")
        bpy.ops.render.render(write_still=True)

        print(f"✅ Render complete! Saved to: {output_path}")

    def run_all_demos(self):
        """Alle Demos ausführen"""
        print("🚀 Starting Astronomy Workflow Demo...")

        # Demo 1: Basic Objects
        basic_objects = self.demo_1_basic_objects()

        # Demo 2: Complex Scene
        complex_objects = self.demo_2_complex_scene()

        # Demo 3: Lighting
        self.demo_3_lighting_setup()

        # Demo 4: Camera
        self.demo_4_camera_setup()

        # Demo 5: Export
        self.demo_5_export_system(complex_objects)

        # Demo 6: Rendering
        self.demo_6_rendering()

        print("\n🎉 All Demos Complete!")
        print(f"📁 Check outputs in: {self.output_folder.absolute()}")


def main():
    """Main function"""

    # Demo-Workflow erstellen
    demo = AstronomyWorkflowDemo(output_folder="astronomy_workflow_demo", scene_seed=42)

    # Alle Demos ausführen
    demo.run_all_demos()


if __name__ == "__main__":
    main()
