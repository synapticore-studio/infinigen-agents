#!/usr/bin/env python3
"""
Advanced Terrain Engine - Vollständige Implementierung mit allen Maps
Nutzt vorhandene Infinigen-Codebase und moderne Packages ohne Redundanz
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from torch_geometric.transforms import NormalizeFeatures

from infinigen.assets.composition import material_assignments
from infinigen.core.tagging import tag_object
from infinigen.core.util.organization import Tags, TerrainNames
from infinigen.terrain.utils import Mesh

# Infinigen Core Imports - Nutze vorhandene Codebase
from infinigen.terrain.utils.image_processing import get_normal, sharpen

logger = logging.getLogger(__name__)


class TerrainMapGenerator:
    """Generiert alle Terrain-Maps (Height, Normal, Displacement, etc.)"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

    def generate_height_map(
        self, terrain_type: str, seed: int, resolution: int = 512
    ) -> np.ndarray:
        """Generiere Heightmap basierend auf Terrain-Typ"""
        np.random.seed(seed)

        if terrain_type == "mountain":
            return self._generate_mountain_heightmap(resolution)
        elif terrain_type == "hills":
            return self._generate_hills_heightmap(resolution)
        elif terrain_type == "valley":
            return self._generate_valley_heightmap(resolution)
        elif terrain_type == "plateau":
            return self._generate_plateau_heightmap(resolution)
        else:
            return self._generate_default_heightmap(resolution)

    def _generate_mountain_heightmap(self, resolution: int) -> np.ndarray:
        """Generiere Berg-Terrain mit mehreren Noise-Layern"""
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        X, Y = np.meshgrid(x, y)

        # Mehrere Perlin-Noise-Layer
        height = np.zeros_like(X)

        # Basis-Terrain
        height += 0.5 * self._perlin_noise(X, Y, scale=0.5)

        # Berg-Formen
        height += 0.3 * self._perlin_noise(X, Y, scale=0.2)

        # Details
        height += 0.1 * self._perlin_noise(X, Y, scale=0.05)

        # Schärfe anwenden
        height = sharpen(height)

        return height

    def _generate_hills_heightmap(self, resolution: int) -> np.ndarray:
        """Generiere Hügel-Terrain"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Sanfte Hügel
        height = 0.3 * self._perlin_noise(X, Y, scale=0.3)
        height += 0.1 * self._perlin_noise(X, Y, scale=0.1)

        return height

    def _generate_valley_heightmap(self, resolution: int) -> np.ndarray:
        """Generiere Tal-Terrain"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Tal-Form
        distance = np.sqrt(X**2 + Y**2)
        valley = -0.5 * np.exp(-(distance**2) / 0.5)

        # Details hinzufügen
        noise = 0.1 * self._perlin_noise(X, Y, scale=0.2)

        return valley + noise

    def _generate_plateau_heightmap(self, resolution: int) -> np.ndarray:
        """Generiere Plateau-Terrain"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Plateau mit Rändern
        plateau = np.ones_like(X) * 0.5
        edge_falloff = np.exp(-(X**2 + Y**2) / 0.3)
        plateau *= edge_falloff

        # Leichte Variation
        plateau += 0.05 * self._perlin_noise(X, Y, scale=0.1)

        return plateau

    def _generate_default_heightmap(self, resolution: int) -> np.ndarray:
        """Standard-Terrain"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        return 0.2 * self._perlin_noise(X, Y, scale=0.3)

    def _perlin_noise(
        self, X: np.ndarray, Y: np.ndarray, scale: float = 1.0
    ) -> np.ndarray:
        """Einfache Perlin-Noise-Implementierung"""
        # Vereinfachte Perlin-Noise mit Sinus-Funktionen
        return np.sin(X * scale) * np.cos(Y * scale) + 0.5 * np.sin(
            X * scale * 2
        ) * np.cos(Y * scale * 2)

    def generate_normal_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generiere Normal Map aus Heightmap - Nutze Infinigen-Funktion"""
        return get_normal(height_map, height_map.shape[0])

    def generate_displacement_map(
        self, height_map: np.ndarray, strength: float = 0.1
    ) -> np.ndarray:
        """Generiere Displacement Map"""
        # Displacement basierend auf Höhenunterschieden
        grad_x = np.gradient(height_map, axis=1)
        grad_y = np.gradient(height_map, axis=0)

        # Displacement-Vektor
        displacement = np.sqrt(grad_x**2 + grad_y**2) * strength
        return displacement

    def generate_roughness_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generiere Roughness Map für Materialien"""
        # Roughness basierend auf lokalen Höhenvariationen
        from scipy.ndimage import gaussian_filter

        # Glatte Version
        smooth = gaussian_filter(height_map, sigma=2.0)

        # Roughness = Differenz zwischen original und glatt
        roughness = np.abs(height_map - smooth)

        # Normalisiere auf 0-1
        roughness = (roughness - roughness.min()) / (
            roughness.max() - roughness.min() + 1e-8
        )

        return roughness

    def generate_ao_map(self, height_map: np.ndarray) -> np.ndarray:
        """Generiere Ambient Occlusion Map"""
        # Vereinfachte AO-Berechnung
        h, w = height_map.shape
        ao = np.ones_like(height_map)

        # Sample um jeden Punkt
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center_height = height_map[i, j]

                # Prüfe Nachbarpunkte
                neighbors = [
                    height_map[i - 1, j - 1],
                    height_map[i - 1, j],
                    height_map[i - 1, j + 1],
                    height_map[i, j - 1],
                    height_map[i, j + 1],
                    height_map[i + 1, j - 1],
                    height_map[i + 1, j],
                    height_map[i + 1, j + 1],
                ]

                # AO = Anteil der höheren Nachbarn
                higher_count = sum(1 for h in neighbors if h > center_height)
                ao[i, j] = 1.0 - (higher_count / len(neighbors)) * 0.5

        return ao


class BlenderTerrainIntegrator:
    """Integriert Terrain in Blender mit allen Maps"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_terrain_mesh(
        self, height_map: np.ndarray, name: str = "terrain"
    ) -> Optional[bpy.types.Object]:
        """Erstelle Blender-Mesh aus Heightmap"""
        try:
            # Alte Meshes löschen
            if name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

            h, w = height_map.shape
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(x, y)

            # Vertices
            vertices = []
            for i in range(h):
                for j in range(w):
                    vertices.append([X[i, j], Y[i, j], height_map[i, j]])

            # Faces
            faces = []
            for i in range(h - 1):
                for j in range(w - 1):
                    v1 = i * w + j
                    v2 = v1 + 1
                    v3 = (i + 1) * w + j
                    v4 = v3 + 1

                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])

            # Mesh erstellen
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            # Objekt erstellen
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)

            # Tagging für Infinigen-Kompatibilität
            tag_object(obj, Tags.Terrain)
            tag_object(obj, Tags.MainTerrain)

            return obj

        except Exception as e:
            self.logger.error(f"Error creating terrain mesh: {e}")
            return None

    def apply_terrain_materials(
        self,
        terrain_mesh: bpy.types.Object,
        terrain_type: str,
        height_map: np.ndarray,
        normal_map: np.ndarray,
        displacement_map: np.ndarray,
        roughness_map: np.ndarray,
        ao_map: np.ndarray,
    ) -> None:
        """Wende Terrain-Materialien mit allen Maps an"""
        try:
            # Material erstellen
            mat_name = f"{terrain_type}_terrain_material"
            if mat_name in bpy.data.materials:
                bpy.data.materials.remove(bpy.data.materials[mat_name])

            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.node_tree.nodes.clear()

            # Nodes erstellen
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Output Node
            output = nodes.new(type="ShaderNodeOutputMaterial")

            # Principled BSDF
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

            # Texture-Koordinaten
            tex_coord = nodes.new(type="ShaderNodeTexCoord")

            # Mapping
            mapping = nodes.new(type="ShaderNodeMapping")

            # Image-Texturen für Maps
            height_tex = self._create_image_texture(
                height_map, f"{terrain_type}_height"
            )
            normal_tex = self._create_image_texture(
                normal_map, f"{terrain_type}_normal"
            )
            roughness_tex = self._create_image_texture(
                roughness_map, f"{terrain_type}_roughness"
            )
            ao_tex = self._create_image_texture(ao_map, f"{terrain_type}_ao")

            # Image Texture Nodes
            height_img = nodes.new(type="ShaderNodeTexImage")
            height_img.image = height_tex

            normal_img = nodes.new(type="ShaderNodeTexImage")
            normal_img.image = normal_tex

            roughness_img = nodes.new(type="ShaderNodeTexImage")
            roughness_img.image = roughness_tex

            ao_img = nodes.new(type="ShaderNodeTexImage")
            ao_img.image = ao_tex

            # Normal Map Node
            normal_map_node = nodes.new(type="ShaderNodeNormalMap")

            # Verbindungen
            links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], height_img.inputs["Vector"])
            links.new(mapping.outputs["Vector"], normal_img.inputs["Vector"])
            links.new(mapping.outputs["Vector"], roughness_img.inputs["Vector"])
            links.new(mapping.outputs["Vector"], ao_img.inputs["Vector"])

            # Normal Map
            links.new(normal_img.outputs["Color"], normal_map_node.inputs["Color"])
            links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

            # Roughness
            links.new(roughness_img.outputs["Color"], bsdf.inputs["Roughness"])

            # AO
            links.new(ao_img.outputs["Color"], bsdf.inputs["Base Color"])

            # Displacement
            displacement = nodes.new(type="ShaderNodeDisplacement")
            links.new(height_img.outputs["Color"], displacement.inputs["Height"])
            links.new(
                displacement.outputs["Displacement"], output.inputs["Displacement"]
            )

            # BSDF zu Output
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

            # Material zu Mesh
            terrain_mesh.data.materials.append(mat)

            self.logger.info(f"✅ Terrain material applied with all maps")

        except Exception as e:
            self.logger.error(f"Error applying terrain materials: {e}")

    def _create_image_texture(self, data: np.ndarray, name: str) -> bpy.types.Image:
        """Erstelle Blender-Image aus NumPy-Array"""
        # Normalisiere Daten auf 0-1
        if data.ndim == 2:
            # Grayscale
            normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
            # Zu RGB konvertieren
            rgb_data = np.stack([normalized] * 3, axis=2)
        else:
            # Bereits RGB
            rgb_data = data

        # Zu 0-255 konvertieren
        rgb_data = (rgb_data * 255).astype(np.uint8)

        # Blender Image erstellen
        h, w = rgb_data.shape[:2]
        image = bpy.data.images.new(name, width=w, height=h)
        image.pixels = rgb_data.flatten()

        return image


class AdvancedTerrainEngine:
    """Vollständige Terrain-Engine mit allen Features"""

    def __init__(self, device: str = "cpu", db_path: Path = Path("terrain.db")):
        self.device = device
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Komponenten
        self.map_generator = TerrainMapGenerator(device)
        self.blender_integrator = BlenderTerrainIntegrator()

        # DuckDB für Speicherung
        self._init_database()

    def _init_database(self):
        """Initialisiere DuckDB-Datenbank"""
        try:
            import duckdb

            self.conn = duckdb.connect(str(self.db_path))

            # Tabelle für Terrain-Maps
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terrain_maps (
                    id INTEGER PRIMARY KEY,
                    terrain_type VARCHAR,
                    seed INTEGER,
                    resolution INTEGER,
                    height_map BLOB,
                    normal_map BLOB,
                    displacement_map BLOB,
                    roughness_map BLOB,
                    ao_map BLOB,
                    generation_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            self.logger.info("✅ Terrain database initialized")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.conn = None

    def generate_terrain(
        self, terrain_type: str, seed: int, resolution: int = 512
    ) -> Dict[str, Any]:
        """Generiere vollständiges Terrain mit allen Maps"""
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating advanced terrain: {terrain_type} (seed: {seed})"
            )

            # 1. Height Map generieren
            height_map = self.map_generator.generate_height_map(
                terrain_type, seed, resolution
            )

            # 2. Alle Maps generieren
            normal_map = self.map_generator.generate_normal_map(height_map)
            displacement_map = self.map_generator.generate_displacement_map(height_map)
            roughness_map = self.map_generator.generate_roughness_map(height_map)
            ao_map = self.map_generator.generate_ao_map(height_map)

            # 3. Blender-Mesh erstellen
            terrain_mesh = self.blender_integrator.create_terrain_mesh(
                height_map, f"{terrain_type}_terrain_{seed}"
            )

            # 4. Materialien mit Maps anwenden
            if terrain_mesh:
                self.blender_integrator.apply_terrain_materials(
                    terrain_mesh,
                    terrain_type,
                    height_map,
                    normal_map,
                    displacement_map,
                    roughness_map,
                    ao_map,
                )

            # 5. In Datenbank speichern
            generation_time = time.time() - start_time
            terrain_id = self._store_terrain_maps(
                terrain_type,
                seed,
                resolution,
                height_map,
                normal_map,
                displacement_map,
                roughness_map,
                ao_map,
                generation_time,
            )

            return {
                "success": True,
                "terrain_mesh": terrain_mesh,
                "terrain_id": terrain_id,
                "height_map": height_map,
                "normal_map": normal_map,
                "displacement_map": displacement_map,
                "roughness_map": roughness_map,
                "ao_map": ao_map,
                "generation_time": generation_time,
                "vertices_count": height_map.size,
                "faces_count": (height_map.shape[0] - 1)
                * (height_map.shape[1] - 1)
                * 2,
                "device": self.device,
            }

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Advanced terrain generation failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "generation_time": generation_time,
                "terrain_type": terrain_type,
                "seed": seed,
            }

    def _store_terrain_maps(
        self,
        terrain_type: str,
        seed: int,
        resolution: int,
        height_map: np.ndarray,
        normal_map: np.ndarray,
        displacement_map: np.ndarray,
        roughness_map: np.ndarray,
        ao_map: np.ndarray,
        generation_time: float,
    ) -> int:
        """Speichere alle Terrain-Maps in Datenbank"""
        try:
            if self.conn is None:
                return -1

            # Nächste ID ermitteln
            result = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM terrain_maps"
            ).fetchone()
            terrain_id = result[0] + 1

            # Maps als BLOB speichern
            import pickle

            self.conn.execute(
                """
                INSERT INTO terrain_maps 
                (id, terrain_type, seed, resolution, height_map, normal_map, 
                 displacement_map, roughness_map, ao_map, generation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    terrain_id,
                    terrain_type,
                    seed,
                    resolution,
                    pickle.dumps(height_map),
                    pickle.dumps(normal_map),
                    pickle.dumps(displacement_map),
                    pickle.dumps(roughness_map),
                    pickle.dumps(ao_map),
                    generation_time,
                ),
            )

            self.logger.info(f"✅ Terrain maps stored with ID: {terrain_id}")
            return terrain_id

        except Exception as e:
            self.logger.error(f"Error storing terrain maps: {e}")
            return -1

    def get_available_terrain_types(self) -> List[str]:
        """Verfügbare Terrain-Typen"""
        return ["mountain", "hills", "valley", "plateau", "default"]

    def cleanup(self):
        """Cleanup-Ressourcen"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
