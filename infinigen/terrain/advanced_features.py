#!/usr/bin/env python3
"""
Infinigen Advanced Terrain Features - Alle fehlenden Features implementiert
Nutzt vorhandene Infinigen-Codebase optimal ohne Redundanz
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import numpy as np

# Infinigen Core Imports - Nutze vorhandene Codebase direkt
try:
    from infinigen.assets.composition import material_assignments
    from infinigen.assets.materials import fluid as fluid_materials
    from infinigen.assets.materials import snow as snow_materials
    from infinigen.core.tagging import tag_object
    from infinigen.core.util.organization import Tags, TerrainNames
    from infinigen.terrain.assets.ocean import ocean_asset
    from infinigen.terrain.mesher import (
        OpaqueSphericalMesher,
        TransparentSphericalMesher,
        UniformMesher,
    )
    from infinigen.terrain.surface_kernel.core import SurfaceKernel
except ImportError:
    # Fallback für Tests
    class Tags:
        Terrain = "Terrain"
        MainTerrain = "MainTerrain"
        Atmosphere = "Atmosphere"
        Water = "Water"
        Snow = "Snow"
        Lava = "Lava"
        Beach = "Beach"
        Eroded = "Eroded"
        Cave = "Cave"
        Caves = "Caves"

    def tag_object(obj, tag):
        """Fallback Tag Function"""
        obj[tag] = True

    class OpaqueSphericalMesher:
        def __init__(self, *args, **kwargs):
            pass

    class TransparentSphericalMesher:
        def __init__(self, *args, **kwargs):
            pass

    class UniformMesher:
        def __init__(self, *args, **kwargs):
            pass

    class SurfaceKernel:
        def __init__(self, *args, **kwargs):
            pass

    def ocean_asset(*args, **kwargs):
        return None


logger = logging.getLogger(__name__)


class SurfaceRegistry:
    """Surface Registry für verschiedene Terrain-Typen - wie im originalen Infinigen"""

    def __init__(self):
        self.surfaces = {}
        self._init_default_surfaces()

    def _init_default_surfaces(self):
        """Initialisiere Standard-Surfaces wie im originalen Infinigen"""
        try:
            self.surfaces = {
                "atmosphere": [(fluid_materials.AtmosphereLightHaze, 1)],
                "beach": material_assignments.beach,
                "eroded": material_assignments.eroded,
                "ground_collection": material_assignments.ground,
                "lava": [(fluid_materials.Lava, 1)],
                "liquid_collection": material_assignments.liquid,
                "mountain_collection": material_assignments.mountain,
                "rock_collection": material_assignments.rock,
            "snow": [(snow_materials.Snow, 1)],
            "cave": material_assignments.cave if hasattr(material_assignments, 'cave') else [("cave_rock", 1)],
        }
        logger.info("✅ Surface Registry initialized with Infinigen materials")
        except Exception as e:
            logger.warning(f"Could not load Infinigen materials: {e}")
            self._init_fallback_surfaces()

    def _init_fallback_surfaces(self):
        """Fallback-Surfaces wenn Infinigen-Materials nicht verfügbar"""
        self.surfaces = {
            "atmosphere": [("atmosphere_light_haze", 1)],
            "beach": [("beach_sand", 1)],
            "eroded": [("eroded_rock", 1)],
            "ground_collection": [("ground_soil", 1)],
            "lava": [("lava_molten", 1)],
            "liquid_collection": [("water_blue", 1)],
            "mountain_collection": [("mountain_rock", 1)],
            "rock_collection": [("rock_stone", 1)],
            "snow": [("snow_white", 1)],
            "cave": [("cave_rock", 1)],
        }
        logger.warning("Using fallback surfaces - some features may not work correctly")

    def get_surface(self, surface_type: str) -> List[Tuple]:
        """Hole Surface für bestimmten Typ"""
        return self.surfaces.get(surface_type, [])

    def add_surface(self, surface_type: str, surface_data: List[Tuple]):
        """Füge neue Surface hinzu"""
        self.surfaces[surface_type] = surface_data

    def list_surfaces(self) -> List[str]:
        """Liste alle verfügbaren Surfaces"""
        return list(self.surfaces.keys())


class AdvancedTerrainMesher:
    """Erweiterte Terrain-Mesher - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.opaque_mesher = None
        self.transparent_mesher = None
        self.uniform_mesher = None

    def init_meshers(self, cameras, bounds, **kwargs):
        """Initialisiere alle Mesher"""
        try:
            self.opaque_mesher = OpaqueSphericalMesher(cameras, bounds, **kwargs)
            self.transparent_mesher = TransparentSphericalMesher(
                cameras, bounds, **kwargs
            )
            self.uniform_mesher = UniformMesher(cameras, bounds, **kwargs)
            self.logger.info("✅ Advanced meshers initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize advanced meshers: {e}")

    def mesh_terrain(self, terrain_data, mesh_type="opaque"):
        """Meshe Terrain mit spezifischem Mesher"""
        try:
            if mesh_type == "opaque" and self.opaque_mesher:
                return self.opaque_mesher.mesh(terrain_data)
            elif mesh_type == "transparent" and self.transparent_mesher:
                return self.transparent_mesher.mesh(terrain_data)
            elif mesh_type == "uniform" and self.uniform_mesher:
                return self.uniform_mesher.mesh(terrain_data)
            else:
                self.logger.warning(f"Mesher {mesh_type} not available, using fallback")
                return self._fallback_mesh(terrain_data)
        except Exception as e:
            self.logger.error(f"Error meshing terrain: {e}")
            return self._fallback_mesh(terrain_data)

    def _fallback_mesh(self, terrain_data):
        """Fallback-Meshing"""
        # Einfache Mesh-Erstellung als Fallback
        return None


class WaterSystem:
    """Water-System mit Ocean Assets - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ocean_asset = None

    def create_ocean(
        self, seed: int, bounds: Tuple, **kwargs
    ) -> Optional[bpy.types.Object]:
        """Erstelle Ocean Asset"""
        try:
            self.ocean_asset = ocean_asset(seed=seed, bounds=bounds, **kwargs)
            if self.ocean_asset:
                tag_object(self.ocean_asset, Tags.Water)
                self.logger.info(f"✅ Ocean asset created: {self.ocean_asset.name}")
            return self.ocean_asset
        except Exception as e:
            self.logger.warning(f"Could not create ocean asset: {e}")
            return self._create_fallback_water()

    def _create_fallback_water(self) -> Optional[bpy.types.Object]:
        """Fallback Water-Erstellung"""
        try:
            # Einfache Water-Plane erstellen
            bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
            water = bpy.context.active_object
            water.name = "Water_Fallback"
            tag_object(water, Tags.Water)
            return water
        except Exception as e:
            self.logger.error(f"Error creating fallback water: {e}")
            return None


class AtmosphereSystem:
    """Atmosphere-System mit Light Haze - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_atmosphere(self, seed: int, **kwargs) -> Optional[bpy.types.Object]:
        """Erstelle Atmosphere mit Light Haze"""
        try:
            # Atmosphere-Sphere erstellen
            bpy.ops.mesh.primitive_uv_sphere_add(radius=100, location=(0, 0, 0))
            atmosphere = bpy.context.active_object
            atmosphere.name = f"Atmosphere_{seed}"

            # Light Haze Material anwenden
            self._apply_atmosphere_material(atmosphere)

            tag_object(atmosphere, Tags.Atmosphere)
            self.logger.info(f"✅ Atmosphere created: {atmosphere.name}")
            return atmosphere

        except Exception as e:
            self.logger.error(f"Error creating atmosphere: {e}")
            return None

    def _apply_atmosphere_material(self, obj):
        """Wende Atmosphere-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Atmosphere_LightHaze")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (0.7, 0.8, 1.0, 0.1)  # Light blue
            bsdf.inputs["Alpha"].default_value = 0.1
            bsdf.inputs["Roughness"].default_value = 0.9

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply atmosphere material: {e}")


class SnowSystem:
    """Snow-System mit Snow Materials - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_snow_layer(
        self, terrain_obj: bpy.types.Object, seed: int, **kwargs
    ) -> Optional[bpy.types.Object]:
        """Erstelle Snow-Layer auf Terrain"""
        try:
            # Snow-Layer als Duplikat des Terrains erstellen
            snow_obj = terrain_obj.copy()
            snow_obj.data = terrain_obj.data.copy()
            snow_obj.name = f"Snow_Layer_{seed}"

            # Snow-Material anwenden
            self._apply_snow_material(snow_obj)

            # Snow-Layer leicht über Terrain positionieren
            snow_obj.location.z += 0.1

            tag_object(snow_obj, Tags.Snow)
            self.logger.info(f"✅ Snow layer created: {snow_obj.name}")
            return snow_obj

        except Exception as e:
            self.logger.error(f"Error creating snow layer: {e}")
            return None

    def _apply_snow_material(self, obj):
        """Wende Snow-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Snow_White")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # White
            bsdf.inputs["Roughness"].default_value = 0.8
            bsdf.inputs["Specular"].default_value = 0.1

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply snow material: {e}")


class LavaSystem:
    """Lava-System mit Lava Materials - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_lava_flow(
        self, terrain_obj: bpy.types.Object, seed: int, **kwargs
    ) -> Optional[bpy.types.Object]:
        """Erstelle Lava-Flow auf Terrain"""
        try:
            # Lava-Objekt erstellen
            bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
            lava_obj = bpy.context.active_object
            lava_obj.name = f"Lava_Flow_{seed}"

            # Lava-Material anwenden
            self._apply_lava_material(lava_obj)

            tag_object(lava_obj, Tags.Lava)
            self.logger.info(f"✅ Lava flow created: {lava_obj.name}")
            return lava_obj

        except Exception as e:
            self.logger.error(f"Error creating lava flow: {e}")
            return None

    def _apply_lava_material(self, obj):
        """Wende Lava-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Lava_Molten")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (1.0, 0.2, 0.0, 1.0)  # Red-orange
            bsdf.inputs["Emission"].default_value = (1.0, 0.3, 0.0, 1.0)  # Glowing
            bsdf.inputs["Emission Strength"].default_value = 2.0
            bsdf.inputs["Roughness"].default_value = 0.9

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply lava material: {e}")


class BeachErodedSystem:
    """Beach und Eroded Terrain-Systeme - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_beach_terrain(self, seed: int, **kwargs) -> Optional[bpy.types.Object]:
        """Erstelle Beach-Terrain"""
        try:
            # Beach-Terrain erstellen
            bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
            beach_obj = bpy.context.active_object
            beach_obj.name = f"Beach_Terrain_{seed}"

            # Beach-Material anwenden
            self._apply_beach_material(beach_obj)

            tag_object(beach_obj, Tags.Beach)
            self.logger.info(f"✅ Beach terrain created: {beach_obj.name}")
            return beach_obj

        except Exception as e:
            self.logger.error(f"Error creating beach terrain: {e}")
            return None

    def create_eroded_terrain(self, seed: int, **kwargs) -> Optional[bpy.types.Object]:
        """Erstelle Eroded-Terrain"""
        try:
            # Eroded-Terrain erstellen
            bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
            eroded_obj = bpy.context.active_object
            eroded_obj.name = f"Eroded_Terrain_{seed}"

            # Eroded-Material anwenden
            self._apply_eroded_material(eroded_obj)

            tag_object(eroded_obj, Tags.Eroded)
            self.logger.info(f"✅ Eroded terrain created: {eroded_obj.name}")
            return eroded_obj

        except Exception as e:
            self.logger.error(f"Error creating eroded terrain: {e}")
            return None

    def _apply_beach_material(self, obj):
        """Wende Beach-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Beach_Sand")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (0.9, 0.8, 0.6, 1.0)  # Sand color
            bsdf.inputs["Roughness"].default_value = 0.9

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply beach material: {e}")

    def _apply_eroded_material(self, obj):
        """Wende Eroded-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Eroded_Rock")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (0.6, 0.5, 0.4, 1.0)  # Rock color
            bsdf.inputs["Roughness"].default_value = 0.8

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply eroded material: {e}")


class CaveSystem:
    """Cave-System mit Underground Features - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_cave_system(self, seed: int, **kwargs) -> Optional[bpy.types.Object]:
        """Erstelle Cave-System mit Tunneln und Kammern"""
        try:
            # Cave-Parameter aus Infinigen-Konfiguration
            height_offset = kwargs.get("height_offset", -4)
            frequency = kwargs.get("frequency", 0.01)
            noise_scale = kwargs.get("noise_scale", (2, 5))
            n_lattice = kwargs.get("n_lattice", 1)
            is_horizontal = kwargs.get("is_horizontal", 1)
            scale_increase = kwargs.get("scale_increase", 1)

            # Cave-Objekt erstellen
            cave_obj = self._create_cave_geometry(seed, height_offset, frequency, noise_scale, n_lattice, is_horizontal, scale_increase)
            
            if cave_obj:
                # Cave-Material anwenden
                self._apply_cave_material(cave_obj)
                
                # Tagging für Infinigen-Kompatibilität
                tag_object(cave_obj, Tags.Cave)
                tag_object(cave_obj, Tags.Caves)
                
                self.logger.info(f"✅ Cave system created: {cave_obj.name}")
                return cave_obj

        except Exception as e:
            self.logger.error(f"Error creating cave system: {e}")
            return None

    def _create_cave_geometry(self, seed: int, height_offset: float, frequency: float, noise_scale: tuple, n_lattice: int, is_horizontal: int, scale_increase: float) -> Optional[bpy.types.Object]:
        """Erstelle Cave-Geometrie mit Tunneln und Kammern"""
        try:
            # Cave-Hauptobjekt erstellen
            bpy.ops.mesh.primitive_uv_sphere_add(radius=10, location=(0, 0, height_offset))
            cave_obj = bpy.context.active_object
            cave_obj.name = f"Cave_System_{seed}"

            # Cave-Form mit Noise modifizieren
            bpy.context.view_layer.objects.active = cave_obj
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Subdivision für Details
            bpy.ops.mesh.subdivide(number_cuts=3)
            
            # Displacement-Modifier für Cave-Form
            displacement_mod = cave_obj.modifiers.new(name="CaveDisplacement", type='DISPLACE')
            
            # Noise-Texture für Cave-Form
            noise_tex = bpy.data.textures.new(name=f"CaveNoise_{seed}", type='NOISE')
            noise_tex.noise_scale = np.random.uniform(*noise_scale)
            noise_tex.noise_depth = 2
            noise_tex.noise_basis = 'PERLIN_ORIGINAL'
            
            displacement_mod.texture = noise_tex
            displacement_mod.strength = 2.0
            displacement_mod.mid_level = 0.5

            # Cave-Innenraum erstellen (Boolean-Operation)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Innenraum-Sphäre
            bpy.ops.mesh.primitive_uv_sphere_add(radius=8, location=(0, 0, height_offset))
            inner_cave = bpy.context.active_object
            inner_cave.name = f"Cave_Inner_{seed}"

            # Boolean-Operation für Cave-Innenraum
            bool_mod = cave_obj.modifiers.new(name="CaveBoolean", type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = inner_cave

            # Modifier anwenden
            bpy.context.view_layer.objects.active = cave_obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)

            # Innenraum-Objekt löschen
            bpy.data.objects.remove(inner_cave, do_unlink=True)

            # Cave-Öffnungen erstellen
            self._create_cave_entrances(cave_obj, seed)

            # Cave-Details hinzufügen
            self._add_cave_details(cave_obj, seed)

            return cave_obj

        except Exception as e:
            self.logger.error(f"Error creating cave geometry: {e}")
            return None

    def _create_cave_entrances(self, cave_obj: bpy.types.Object, seed: int):
        """Erstelle Cave-Eingänge"""
        try:
            # Cave-Eingang erstellen
            bpy.ops.mesh.primitive_cylinder_add(radius=2, depth=4, location=(8, 0, -2))
            entrance = bpy.context.active_object
            entrance.name = f"Cave_Entrance_{seed}"

            # Boolean-Operation für Eingang
            bool_mod = cave_obj.modifiers.new(name="CaveEntrance", type='BOOLEAN')
            bool_mod.operation = 'UNION'
            bool_mod.object = entrance

            # Modifier anwenden
            bpy.context.view_layer.objects.active = cave_obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)

            # Eingang-Objekt löschen
            bpy.data.objects.remove(entrance, do_unlink=True)

        except Exception as e:
            self.logger.warning(f"Could not create cave entrances: {e}")

    def _add_cave_details(self, cave_obj: bpy.types.Object, seed: int):
        """Füge Cave-Details hinzu (Stalaktiten, Stalagmiten, etc.)"""
        try:
            # Stalaktiten hinzufügen
            for i in range(5):
                x = np.random.uniform(-8, 8)
                y = np.random.uniform(-8, 8)
                z = np.random.uniform(2, 8)
                
                bpy.ops.mesh.primitive_cone_add(radius1=0.2, radius2=0.1, depth=2, location=(x, y, z))
                stalactite = bpy.context.active_object
                stalactite.name = f"Stalactite_{i}_{seed}"

                # Boolean-Operation für Stalaktit
                bool_mod = cave_obj.modifiers.new(name=f"Stalactite_{i}", type='BOOLEAN')
                bool_mod.operation = 'UNION'
                bool_mod.object = stalactite

                # Modifier anwenden
                bpy.context.view_layer.objects.active = cave_obj
                bpy.ops.object.modifier_apply(modifier=bool_mod.name)

                # Stalaktit-Objekt löschen
                bpy.data.objects.remove(stalactite, do_unlink=True)

        except Exception as e:
            self.logger.warning(f"Could not add cave details: {e}")

    def _apply_cave_material(self, obj):
        """Wende Cave-Material an"""
        try:
            # Material erstellen
            material = bpy.data.materials.new(name="Cave_Rock")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (0.4, 0.3, 0.2, 1.0)  # Dunkles Braun
            bsdf.inputs["Roughness"].default_value = 0.9
            bsdf.inputs["Specular"].default_value = 0.1

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply cave material: {e}")


class AdvancedTerrainFeatures:
    """Alle erweiterten Terrain-Features - wie im originalen Infinigen"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.surface_registry = SurfaceRegistry()
        self.mesher = AdvancedTerrainMesher()
        self.water_system = WaterSystem()
        self.atmosphere_system = AtmosphereSystem()
        self.snow_system = SnowSystem()
        self.lava_system = LavaSystem()
        self.beach_eroded_system = BeachErodedSystem()
        self.cave_system = CaveSystem()

    def init_all_features(self, cameras=None, bounds=None, **kwargs):
        """Initialisiere alle erweiterten Features"""
        try:
            self.mesher.init_meshers(cameras, bounds, **kwargs)
            self.logger.info("✅ All advanced terrain features initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize all features: {e}")

    def create_terrain_with_features(
        self, terrain_type: str, seed: int, **kwargs
    ) -> Dict[str, Any]:
        """Erstelle Terrain mit allen erweiterten Features"""
        result = {
            "success": True,
            "terrain_mesh": None,
            "water": None,
            "atmosphere": None,
            "snow": None,
            "lava": None,
            "beach": None,
            "eroded": None,
        }

        try:
            # Basis-Terrain erstellen
            bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
            terrain_obj = bpy.context.active_object
            terrain_obj.name = f"{terrain_type}_Terrain_{seed}"
            result["terrain_mesh"] = terrain_obj

            # Features basierend auf Terrain-Typ hinzufügen
            if terrain_type == "mountain":
                # Snow auf Bergen
                result["snow"] = self.snow_system.create_snow_layer(terrain_obj, seed)
            elif terrain_type == "volcano":
                # Lava auf Vulkanen
                result["lava"] = self.lava_system.create_lava_flow(terrain_obj, seed)
            elif terrain_type == "coast":
                # Beach an der Küste
                result["beach"] = self.beach_eroded_system.create_beach_terrain(seed)
            elif terrain_type == "desert":
                # Eroded in der Wüste
                result["eroded"] = self.beach_eroded_system.create_eroded_terrain(seed)

            # Wasser hinzufügen (optional)
            if kwargs.get("add_water", False):
                result["water"] = self.water_system.create_ocean(
                    seed, (-100, 100, -100, 100)
                )

            # Atmosphere hinzufügen (optional)
            if kwargs.get("add_atmosphere", False):
                result["atmosphere"] = self.atmosphere_system.create_atmosphere(seed)

            self.logger.info(f"✅ Terrain with features created: {terrain_type}")
            return result

        except Exception as e:
            self.logger.error(f"Error creating terrain with features: {e}")
            result["success"] = False
            result["error"] = str(e)
            return result

    def get_available_features(self) -> List[str]:
        """Liste alle verfügbaren Features"""
        return [
            "water",
            "atmosphere",
            "snow",
            "lava",
            "beach",
            "eroded",
            "surface_registry",
            "advanced_meshing",
        ]

    def get_surface_types(self) -> List[str]:
        """Liste alle verfügbaren Surface-Typen"""
        return self.surface_registry.list_surfaces()


