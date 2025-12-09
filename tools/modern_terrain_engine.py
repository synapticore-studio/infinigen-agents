#!/usr/bin/env python3
"""Moderne Terrain-Engine mit PyTorch Geometric + Kernels + bpy + DuckDB"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric
from kernels import get_kernel
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from torch_geometric.transforms import SIGN

from deps.knowledge_deps import KnowledgeBaseDep

logger = logging.getLogger(__name__)


class TerrainGraphGenerator:
    """Graph-basierte Terrain-Generierung mit PyTorch Geometric"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.gnn = GraphSAGE(
            in_channels=3, hidden_channels=64, out_channels=1, num_layers=3
        )
        self.transform = SIGN(K=3)  # Multi-scale graph features

    def heightmap_to_vertices(self, height_map: np.ndarray) -> torch.Tensor:
        """Konvertiere Heightmap zu Vertex-Positionen"""
        h, w = height_map.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)

        vertices = np.column_stack([X.ravel(), Y.ravel(), height_map.ravel()])

        return torch.tensor(vertices, dtype=torch.float32, device=self.device)

    def generate_terrain_edges(self, vertices: torch.Tensor) -> torch.Tensor:
        """Generiere Edges für Terrain-Graph basierend auf räumlicher Nähe"""
        # Einfache Grid-basierte Verbindungen statt KNN
        num_vertices = vertices.size(0)
        edges = []

        # Verbinde benachbarte Vertices in einem Grid
        h = int(np.sqrt(num_vertices))
        w = h

        for i in range(h):
            for j in range(w):
                idx = i * w + j

                # Verbinde mit rechten Nachbarn
                if j < w - 1:
                    edges.append([idx, idx + 1])
                    edges.append([idx + 1, idx])

                # Verbinde mit unteren Nachbarn
                if i < h - 1:
                    edges.append([idx, idx + w])
                    edges.append([idx + w, idx])

                # Verbinde mit diagonalen Nachbarn
                if i < h - 1 and j < w - 1:
                    edges.append([idx, idx + w + 1])
                    edges.append([idx + w + 1, idx])

        if edges:
            edge_index = (
                torch.tensor(edges, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )
        else:
            # Fallback: Verbinde alle mit allen (vollständiger Graph)
            edge_index = (
                torch.combinations(torch.arange(num_vertices, device=self.device), 2)
                .t()
                .contiguous()
            )
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        return edge_index

    def generate_terrain_graph(self, height_map: np.ndarray) -> Data:
        """Generiere Terrain als Graph mit GNN"""

        # Konvertiere Heightmap zu Graph
        vertices = self.heightmap_to_vertices(height_map)
        edges = self.generate_terrain_edges(vertices)

        # Graph-Transformation für bessere Features
        graph_data = Data(x=vertices, edge_index=edges)
        graph_data = self.transform(graph_data)

        # GNN-basierte Terrain-Verfeinerung
        with torch.no_grad():
            terrain_heights = self.gnn(graph_data.x, graph_data.edge_index)

        # Konvertiere zurück zu Heightmap
        refined_height_map = terrain_heights.cpu().numpy().reshape(height_map.shape)

        return Data(
            x=vertices,
            edge_index=edges,
            height_map=refined_height_map,
            original_height_map=height_map,
        )


class KernelTerrainInterpolator:
    """Kernel-basierte Terrain-Interpolation mit HuggingFace Kernels"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # HuggingFace Kernels werden dynamisch geladen
        self.terrain_kernel = None

    def get_control_points(
        self, terrain_type: str, seed: int, num_points: int = 20
    ) -> np.ndarray:
        """Generiere Kontrollpunkte basierend auf Terrain-Typ"""
        np.random.seed(seed)

        if terrain_type == "mountain":
            # Konzentrierte Punkte für Gipfel
            points = np.random.normal(0, 0.3, (num_points, 2))
            heights = np.random.exponential(2, num_points)
        elif terrain_type == "desert":
            # Gleichmäßig verteilte Punkte für Dünen
            points = np.random.uniform(-1, 1, (num_points, 2))
            heights = np.random.normal(0.5, 0.3, num_points)
        elif terrain_type == "valley":
            # Täler-Form
            points = np.random.uniform(-1, 1, (num_points, 2))
            heights = -np.abs(points[:, 0]) * 0.5 + np.random.normal(0, 0.2, num_points)
        else:
            # Standard-Verteilung
            points = np.random.uniform(-1, 1, (num_points, 2))
            heights = np.random.normal(0, 0.5, num_points)

        return np.column_stack([points, heights])

    def interpolate_terrain(
        self,
        control_points: np.ndarray,
        terrain_type: str = "mountain",
        grid_size: int = 512,
    ) -> np.ndarray:
        """Kernel-basierte Terrain-Interpolation mit HuggingFace Kernels"""
        try:
            # Versuche HuggingFace Kernel zu laden
            if self.terrain_kernel is None:
                try:
                    # Lade einen optimierten Interpolations-Kernel
                    self.terrain_kernel = get_kernel("kernels-community/interpolation")
                    self.logger.info("✅ HuggingFace Kernel geladen")
                except Exception as e:
                    self.logger.warning(f"⚠️ HuggingFace Kernel nicht verfügbar: {e}")
                    self.logger.info("🔄 Verwende Scipy-Fallback für Interpolation")
                    self.terrain_kernel = None

            if self.terrain_kernel is not None:
                # Verwende HuggingFace Kernel für Interpolation
                x = np.linspace(-1, 1, grid_size)
                y = np.linspace(-1, 1, grid_size)
                X, Y = np.meshgrid(x, y)
                grid_points = np.column_stack([X.ravel(), Y.ravel()])

                # Kernel-basierte Interpolation
                heights = self.terrain_kernel.interpolate(
                    control_points[:, :2], control_points[:, 2], grid_points
                )
                return heights.reshape(grid_size, grid_size)
            else:
                # Fallback: Scipy-basierte Interpolation
                return self._scipy_interpolation(control_points, grid_size)

        except Exception as e:
            self.logger.warning(f"Kernel-Interpolation fehlgeschlagen: {e}")
            # Fallback: Einfache lineare Interpolation
            return self._scipy_interpolation(control_points, grid_size)

    def _scipy_interpolation(
        self, control_points: np.ndarray, grid_size: int
    ) -> np.ndarray:
        """Fallback: Scipy-basierte Interpolation"""
        try:
            from scipy.interpolate import griddata

            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(x, y)

            heights = griddata(
                control_points[:, :2],
                control_points[:, 2],
                (X, Y),
                method="cubic",
                fill_value=0.0,
            )

            return heights
        except ImportError:
            # Letzter Fallback: Einfache RBF-ähnliche Interpolation
            return self._simple_rbf_interpolation(control_points, grid_size)

    def _simple_rbf_interpolation(
        self, control_points: np.ndarray, grid_size: int
    ) -> np.ndarray:
        """Einfache RBF-ähnliche Interpolation ohne externe Dependencies"""
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        heights = np.zeros(len(grid_points))

        for i, (px, py) in enumerate(grid_points):
            # RBF mit Gauß-Kernel
            distances = np.linalg.norm(control_points[:, :2] - [px, py], axis=1)
            weights = np.exp(-(distances**2) / 0.1)  # Sigma = 0.1
            weights = weights / (np.sum(weights) + 1e-8)  # Normalisierung
            heights[i] = np.sum(weights * control_points[:, 2])

        return heights.reshape(grid_size, grid_size)


class BlenderTerrainIntegrator:
    """Blender-Integration für Terrain-Meshes"""

    def __init__(self):
        try:
            import bmesh
            import bpy
            from mathutils import Vector

            self.bpy = bpy
            self.bmesh = bmesh
            self.Vector = Vector
            self.available = True
        except ImportError:
            logger.warning(
                "Blender (bpy) nicht verfügbar - Terrain-Integration deaktiviert"
            )
            self.available = False

    def create_terrain_mesh(
        self, height_map: np.ndarray, name: str = "Terrain"
    ) -> Optional[Any]:
        """Erstelle Blender-Mesh aus Heightmap"""
        if not self.available:
            logger.warning("Blender nicht verfügbar - kann kein Mesh erstellen")
            return None

        try:
            # Erstelle neues Mesh
            mesh = self.bpy.data.meshes.new(name)
            obj = self.bpy.data.objects.new(name, mesh)

            # Konvertiere zu Blender-Format
            bm = self.bmesh.new()

            h, w = height_map.shape
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(x, y)

            # Füge Vertices hinzu
            for i in range(h):
                for j in range(w):
                    vertex = self.Vector((X[i, j], Y[i, j], height_map[i, j]))
                    bm.verts.new(vertex)

            bm.verts.ensure_lookup_table()

            # Füge Faces hinzu
            for i in range(h - 1):
                for j in range(w - 1):
                    v1 = i * w + j
                    v2 = v1 + 1
                    v3 = (i + 1) * w + j
                    v4 = v3 + 1

                    try:
                        bm.faces.new([bm.verts[v1], bm.verts[v2], bm.verts[v3]])
                        bm.faces.new([bm.verts[v2], bm.verts[v4], bm.verts[v3]])
                    except ValueError:
                        # Skip invalid faces
                        continue

            # Update Mesh
            bm.to_mesh(mesh)
            bm.free()

            # Füge zur Szene hinzu
            self.bpy.context.collection.objects.link(obj)

            return obj

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Blender-Meshes: {e}")
            return None

    def apply_terrain_materials(
        self, obj: Any, terrain_type: str = "mountain"
    ) -> Optional[Any]:
        """Wende passende Materialien basierend auf Terrain-Typ an"""
        if not self.available or obj is None:
            return None

        try:
            # Erstelle Material basierend auf Terrain-Typ
            material = self.bpy.data.materials.new(name=f"{terrain_type}_material")
            material.use_nodes = True

            # Shader-Nodes konfigurieren
            nodes = material.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")

            if terrain_type == "mountain":
                bsdf.inputs["Base Color"].default_value = (0.5, 0.4, 0.3, 1.0)  # Braun
                bsdf.inputs["Roughness"].default_value = 0.8
            elif terrain_type == "desert":
                bsdf.inputs["Base Color"].default_value = (0.8, 0.7, 0.4, 1.0)  # Sand
                bsdf.inputs["Roughness"].default_value = 0.9
            elif terrain_type == "valley":
                bsdf.inputs["Base Color"].default_value = (0.3, 0.6, 0.3, 1.0)  # Grün
                bsdf.inputs["Roughness"].default_value = 0.6

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

            return material

        except Exception as e:
            logger.error(f"Fehler beim Anwenden der Materialien: {e}")
            return None


class TerrainDatabase:
    """DuckDB-basierte Terrain-Datenbank mit VSS und Spatial Extensions"""

    def __init__(self, db_path: Path = Path("terrain.db")):
        import duckdb

        self.conn = duckdb.connect(str(db_path))
        self._init_extensions()
        self._init_tables()

    def _init_extensions(self):
        """Lade VSS und Spatial Extensions"""
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            self.conn.execute("INSTALL spatial")
            self.conn.execute("LOAD spatial")
            logger.info("✅ VSS und Spatial Extensions geladen")
        except Exception as e:
            logger.warning(f"⚠️ Extensions konnten nicht geladen werden: {e}")

    def _init_tables(self):
        """Initialisiere Terrain-Datenbank-Tabellen"""

        # Terrain-Metadaten
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terrain_metadata (
                id INTEGER PRIMARY KEY,
                terrain_type VARCHAR,
                seed INTEGER,
                resolution INTEGER,
                vertices_count INTEGER,
                faces_count INTEGER,
                generation_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]
            )
        """
        )

        # Terrain-Performance-Metriken
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terrain_performance (
                id INTEGER PRIMARY KEY,
                terrain_id INTEGER,
                generation_method VARCHAR,
                gpu_memory_used FLOAT,
                cpu_time FLOAT,
                gpu_time FLOAT,
                mesh_quality_score FLOAT,
                FOREIGN KEY (terrain_id) REFERENCES terrain_metadata(id)
            )
        """
        )

    def store_terrain(
        self, terrain_data: Dict[str, Any], height_map: np.ndarray
    ) -> int:
        """Speichere Terrain in Datenbank"""

        # Terrain-Metadaten speichern
        # Generiere ID manuell
        max_id_result = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM terrain_metadata"
        ).fetchone()
        terrain_id = max_id_result[0] + 1 if max_id_result else 1

        self.conn.execute(
            """
            INSERT INTO terrain_metadata 
            (id, terrain_type, seed, resolution, vertices_count, faces_count, generation_time, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                terrain_id,
                terrain_data["terrain_type"],
                terrain_data["seed"],
                terrain_data["resolution"],
                terrain_data.get("vertices_count", height_map.size),
                terrain_data.get(
                    "faces_count",
                    (height_map.shape[0] - 1) * (height_map.shape[1] - 1) * 2,
                ),
                terrain_data["generation_time"],
                terrain_data["embedding"],
            ],
        )

        return terrain_id

    def search_similar_terrain(self, query: str, limit: int = 10) -> List[Tuple]:
        """Semantische Suche nach ähnlichen Terrain"""

        # Hier würde normalerweise ein Embedding-Model verwendet werden
        # Für Demo verwenden wir einen einfachen Vektor
        query_embedding = np.random.rand(384).tolist()

        results = self.conn.execute(
            """
            SELECT t.*, 
                   array_cosine_similarity(t.embedding, ?::FLOAT[384]) as similarity
            FROM terrain_metadata t
            WHERE array_cosine_similarity(t.embedding, ?::FLOAT[384]) > 0.7
            ORDER BY similarity DESC
            LIMIT ?
        """,
            [query_embedding, query_embedding, limit],
        ).fetchall()

        return results


class ModernTerrainEngine:
    """Integrierte Moderne Terrain-Engine"""

    def __init__(self, device: str = "cuda", db_path: Path = Path("terrain.db")):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Komponenten initialisieren
        self.graph_generator = TerrainGraphGenerator(self.device)
        self.kernel_interpolator = KernelTerrainInterpolator()
        self.blender_integrator = BlenderTerrainIntegrator()
        self.database = TerrainDatabase(db_path)

        # Embedding-Model für semantische Suche
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "SentenceTransformers nicht verfügbar - Embeddings deaktiviert"
            )
            self.embedding_model = None

    def generate_terrain(
        self, terrain_type: str, seed: int, resolution: int = 512
    ) -> Dict[str, Any]:
        """Generiere Terrain mit allen Komponenten"""

        start_time = time.time()

        try:
            # 1. Kernel-basierte Terrain-Generierung
            control_points = self.kernel_interpolator.get_control_points(
                terrain_type, seed
            )
            height_map = self.kernel_interpolator.interpolate_terrain(
                control_points, terrain_type, resolution
            )

            # 2. PyTorch Geometric für Graph-basierte Verfeinerung
            terrain_graph = self.graph_generator.generate_terrain_graph(height_map)
            refined_height_map = terrain_graph.height_map

            # 3. Blender-Integration
            terrain_mesh = self.blender_integrator.create_terrain_mesh(
                refined_height_map, f"{terrain_type}_terrain_{seed}"
            )

            # 4. Materialien anwenden
            if terrain_mesh:
                self.blender_integrator.apply_terrain_materials(
                    terrain_mesh, terrain_type
                )

            # 5. In Datenbank speichern
            generation_time = time.time() - start_time

            # Embedding für semantische Suche generieren
            if self.embedding_model:
                embedding = self.embedding_model.encode(
                    f"{terrain_type} terrain {seed}"
                ).tolist()
            else:
                embedding = np.random.rand(384).tolist()

            terrain_data = {
                "terrain_type": terrain_type,
                "seed": seed,
                "resolution": resolution,
                "generation_time": generation_time,
                "embedding": embedding,
                "vertices_count": height_map.size,
                "faces_count": (height_map.shape[0] - 1)
                * (height_map.shape[1] - 1)
                * 2,
            }

            terrain_id = self.database.store_terrain(terrain_data, refined_height_map)

            return {
                "success": True,
                "terrain_mesh": terrain_mesh,
                "terrain_id": terrain_id,
                "generation_time": generation_time,
                "height_map": refined_height_map,
                "vertices_count": height_map.size,
                "faces_count": (height_map.shape[0] - 1)
                * (height_map.shape[1] - 1)
                * 2,
                "device": self.device,
            }

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Terrain-Generierung fehlgeschlagen: {e}")

            return {
                "success": False,
                "error": str(e),
                "generation_time": generation_time,
                "terrain_type": terrain_type,
                "seed": seed,
            }

    def search_terrain(self, query: str) -> List[Tuple]:
        """Suche nach ähnlichen Terrain"""
        return self.database.search_similar_terrain(query)

    def get_available_terrain_types(self) -> List[str]:
        """Gibt verfügbare Terrain-Typen zurück"""
        return ["mountain", "valley", "hills", "plains", "desert", "canyon"]

    def get_terrain_info(self, terrain_id: int) -> Optional[Dict[str, Any]]:
        """Hole Terrain-Informationen aus der Datenbank"""
        try:
            result = self.database.conn.execute(
                """
                SELECT * FROM terrain_metadata WHERE id = ?
            """,
                [terrain_id],
            ).fetchone()

            if result:
                return {
                    "id": result[0],
                    "terrain_type": result[1],
                    "seed": result[2],
                    "resolution": result[3],
                    "vertices_count": result[4],
                    "faces_count": result[5],
                    "generation_time": result[6],
                    "created_at": result[7],
                }
            return None

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Terrain-Informationen: {e}")
            return None


# Dependency Injection
ModernTerrainEngineDep = ModernTerrainEngine
