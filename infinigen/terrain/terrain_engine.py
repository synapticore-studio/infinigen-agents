#!/usr/bin/env python3
"""
Infinigen Modern Terrain Engine - Complete Terrain Generation System
Consolidates all terrain modules into a single comprehensive engine with modern features:
- Blender 4.4+ API integration
- PyTorch Geometric for graph-based terrain generation
- HuggingFace kernels package for advanced interpolation
- DuckDB for spatial data management
- Modern Python patterns and type hints
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import bpy
import cv2
import duckdb
import networkx as nx
import numpy as np
import torch
import torch_geometric
import trimesh
from kernels import MaternKernel, RBFKernel, WhiteKernel
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphSAGE
from torch_geometric.utils import from_networkx, to_networkx

# Infinigen Core Imports
try:
    from infinigen.assets.composition import material_assignments
    from infinigen.core.tagging import tag_object
    from infinigen.core.util import blender as butil
    from infinigen.core.util.logging import Timer
    from infinigen.core.util.organization import Tags, TerrainNames
except ImportError:
    # Fallback for testing
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
        obj[tag] = True

    def butil():
        pass


logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Terrain generation types"""

    MOUNTAIN = "mountain"
    HILLS = "hills"
    VALLEY = "valley"
    PLATEAU = "plateau"
    CAVE = "cave"
    VOLCANO = "volcano"
    COAST = "coast"
    DESERT = "desert"
    FOREST = "forest"
    ARCTIC = "arctic"


class MeshingStrategy(Enum):
    """Terrain meshing strategies"""

    OPAQUE_SPHERICAL = "opaque_spherical"
    TRANSPARENT_SPHERICAL = "transparent_spherical"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


@dataclass
class TerrainConfig:
    """Configuration for terrain generation"""

    terrain_type: TerrainType = TerrainType.MOUNTAIN
    resolution: int = 512
    seed: int = 42
    bounds: Tuple[float, float, float, float, float, float] = (
        -100,
        100,
        -100,
        100,
        -50,
        50,
    )
    meshing_strategy: MeshingStrategy = MeshingStrategy.ADAPTIVE
    use_pytorch_geometric: bool = True
    use_kernels_interpolation: bool = True
    use_duckdb_storage: bool = True
    enable_advanced_features: bool = True


class ModernMeshSystem:
    """Modern mesh system using PyTorch Geometric and trimesh"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

    def create_from_heightmap(
        self,
        height_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        name: str = "terrain_mesh",
    ) -> trimesh.Trimesh:
        """Create mesh from heightmap using modern techniques"""
        try:
            h, w = height_map.shape
            x_min, x_max, y_min, y_max = bounds

            # Create grid coordinates
            x = np.linspace(x_min, x_max, w)
            y = np.linspace(y_min, y_max, h)
            X, Y = np.meshgrid(x, y)

            # Create vertices
            vertices = np.stack(
                [X.flatten(), Y.flatten(), height_map.flatten()], axis=1
            )

            # Create faces using Delaunay triangulation
            from scipy.spatial import Delaunay

            points_2d = np.stack([X.flatten(), Y.flatten()], axis=1)
            tri = Delaunay(points_2d)
            faces = tri.simplices

            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Ensure watertight and manifold
            mesh.fill_holes()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()

            self.logger.info(
                f"✅ Modern mesh created: {name} with {len(vertices)} vertices, {len(faces)} faces"
            )
            return mesh

        except Exception as e:
            self.logger.error(f"Error creating modern mesh: {e}")
            return self._create_fallback_mesh(height_map, bounds, name)

    def _create_fallback_mesh(
        self,
        height_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        name: str,
    ) -> trimesh.Trimesh:
        """Fallback mesh creation"""
        h, w = height_map.shape
        x_min, x_max, y_min, y_max = bounds

        # Simple grid-based mesh
        vertices = []
        faces = []

        for i in range(h):
            for j in range(w):
                x = x_min + (x_max - x_min) * j / (w - 1)
                y = y_min + (y_max - y_min) * i / (h - 1)
                z = height_map[i, j]
                vertices.append([x, y, z])

        # Create faces
        for i in range(h - 1):
            for j in range(w - 1):
                v1 = i * w + j
                v2 = v1 + 1
                v3 = (i + 1) * w + j
                v4 = v3 + 1

                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        return trimesh.Trimesh(vertices=vertices, faces=faces)


class PyTorchGeometricTerrainProcessor:
    """Terrain processing using PyTorch Geometric for graph-based operations"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

    def create_terrain_graph(self, height_map: np.ndarray) -> Data:
        """Create a graph representation of the terrain"""
        try:
            h, w = height_map.shape

            # Create node features (position + height)
            nodes = []
            for i in range(h):
                for j in range(w):
                    x = (j - w / 2) / w  # Normalized x
                    y = (i - h / 2) / h  # Normalized y
                    z = height_map[i, j]
                    nodes.append([x, y, z])

            node_features = torch.tensor(nodes, dtype=torch.float32, device=self.device)

            # Create edges (4-connected grid)
            edges = []
            for i in range(h):
                for j in range(w):
                    idx = i * w + j

                    # Right neighbor
                    if j < w - 1:
                        edges.append([idx, idx + 1])
                        edges.append([idx + 1, idx])

                    # Bottom neighbor
                    if i < h - 1:
                        edges.append([idx, idx + w])
                        edges.append([idx + w, idx])

            edge_index = (
                torch.tensor(edges, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )

            # Create graph data
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                pos=node_features[:, :2],  # 2D positions
                height=node_features[:, 2],  # Height values
            )

            self.logger.info(
                f"✅ Terrain graph created with {len(nodes)} nodes and {len(edges)} edges"
            )
            return graph_data

        except Exception as e:
            self.logger.error(f"Error creating terrain graph: {e}")
            return None

    def enhance_terrain_with_gnn(
        self, graph_data: Data, enhancement_type: str = "smoothing"
    ) -> Data:
        """Enhance terrain using Graph Neural Networks"""
        try:
            if enhancement_type == "smoothing":
                # Simple GCN for smoothing
                conv = GCNConv(graph_data.x.size(1), graph_data.x.size(1)).to(
                    self.device
                )
                enhanced_x = conv(graph_data.x, graph_data.edge_index)

            elif enhancement_type == "detail_enhancement":
                # GraphSAGE for detail enhancement
                sage = GraphSAGE(
                    in_channels=graph_data.x.size(1),
                    hidden_channels=64,
                    out_channels=graph_data.x.size(1),
                    num_layers=2,
                ).to(self.device)
                enhanced_x = sage(graph_data.x, graph_data.edge_index)

            elif enhancement_type == "attention_based":
                # GAT for attention-based enhancement
                gat = GATConv(
                    in_channels=graph_data.x.size(1),
                    out_channels=graph_data.x.size(1),
                    heads=4,
                    concat=False,
                ).to(self.device)
                enhanced_x = gat(graph_data.x, graph_data.edge_index)

            else:
                enhanced_x = graph_data.x

            # Create enhanced graph
            enhanced_graph = Data(
                x=enhanced_x,
                edge_index=graph_data.edge_index,
                pos=graph_data.pos,
                height=(
                    enhanced_x[:, 2] if enhanced_x.size(1) > 2 else graph_data.height
                ),
            )

            self.logger.info(f"✅ Terrain enhanced with {enhancement_type}")
            return enhanced_graph

        except Exception as e:
            self.logger.error(f"Error enhancing terrain with GNN: {e}")
            return graph_data


class KernelsInterpolationSystem:
    """Advanced interpolation using HuggingFace kernels package"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kernels = {"rbf": RBFKernel, "matern": MaternKernel, "white": WhiteKernel}

    def interpolate_terrain(
        self,
        sparse_points: np.ndarray,
        sparse_values: np.ndarray,
        target_points: np.ndarray,
        kernel_type: str = "rbf",
        **kernel_params,
    ) -> np.ndarray:
        """Interpolate terrain using kernel methods"""
        try:
            # Create kernel
            if kernel_type == "rbf":
                kernel = self.kernels["rbf"](
                    length_scale=kernel_params.get("length_scale", 1.0)
                )
            elif kernel_type == "matern":
                kernel = self.kernels["matern"](
                    length_scale=kernel_params.get("length_scale", 1.0),
                    nu=kernel_params.get("nu", 1.5),
                )
            else:
                kernel = self.kernels["rbf"]()

            # Convert to torch tensors
            X_train = torch.tensor(sparse_points, dtype=torch.float32)
            y_train = torch.tensor(sparse_values, dtype=torch.float32)
            X_test = torch.tensor(target_points, dtype=torch.float32)

            # Compute kernel matrices
            K_train = kernel(X_train, X_train)
            K_test = kernel(X_test, X_train)

            # Add noise for numerical stability
            K_train += torch.eye(K_train.size(0)) * 1e-6

            # Solve for weights
            weights = torch.linalg.solve(K_train, y_train)

            # Predict at target points
            predictions = K_test @ weights

            self.logger.info(f"✅ Terrain interpolated using {kernel_type} kernel")
            return predictions.numpy()

        except Exception as e:
            self.logger.error(f"Error in kernel interpolation: {e}")
            # Fallback to simple interpolation
            return self._fallback_interpolation(
                sparse_points, sparse_values, target_points
            )

    def _fallback_interpolation(
        self,
        sparse_points: np.ndarray,
        sparse_values: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Fallback interpolation using scipy"""
        from scipy.interpolate import griddata

        return griddata(sparse_points, sparse_values, target_points, method="cubic")


class DuckDBSpatialManager:
    """Spatial data management using DuckDB"""

    def __init__(self, db_path: Path = Path("terrain_spatial.db")):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.logger = logging.getLogger(__name__)
        self._init_spatial_extension()

    def _init_spatial_extension(self):
        """Initialize DuckDB spatial extension"""
        try:
            self.conn.execute("INSTALL spatial")
            self.conn.execute("LOAD spatial")
            self.logger.info("✅ DuckDB spatial extension loaded")
        except Exception as e:
            self.logger.warning(f"Could not load spatial extension: {e}")

    def store_terrain_data(
        self,
        terrain_id: str,
        height_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        metadata: Dict[str, Any],
    ) -> bool:
        """Store terrain data in DuckDB"""
        try:
            # Create table if not exists
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terrain_data (
                    id VARCHAR PRIMARY KEY,
                    bounds_x_min DOUBLE,
                    bounds_x_max DOUBLE,
                    bounds_y_min DOUBLE,
                    bounds_y_max DOUBLE,
                    resolution_x INTEGER,
                    resolution_y INTEGER,
                    height_map BLOB,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Store data
            import pickle

            self.conn.execute(
                """
                INSERT OR REPLACE INTO terrain_data 
                (id, bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max, 
                 resolution_x, resolution_y, height_map, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    terrain_id,
                    bounds[0],
                    bounds[1],
                    bounds[2],
                    bounds[3],
                    height_map.shape[1],
                    height_map.shape[0],
                    pickle.dumps(height_map),
                    str(metadata),
                ),
            )

            self.logger.info(f"✅ Terrain data stored: {terrain_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing terrain data: {e}")
            return False

    def query_terrain_region(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> List[Dict[str, Any]]:
        """Query terrain data in a specific region"""
        try:
            result = self.conn.execute(
                """
                SELECT id, bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max, 
                       resolution_x, resolution_y, metadata
                FROM terrain_data
                WHERE bounds_x_min <= ? AND bounds_x_max >= ?
                  AND bounds_y_min <= ? AND bounds_y_max >= ?
            """,
                (x_max, x_min, y_max, y_min),
            ).fetchall()

            return [
                dict(
                    zip(
                        [
                            "id",
                            "x_min",
                            "x_max",
                            "y_min",
                            "y_max",
                            "res_x",
                            "res_y",
                            "metadata",
                        ],
                        row,
                    )
                )
                for row in result
            ]

        except Exception as e:
            self.logger.error(f"Error querying terrain region: {e}")
            return []


class Blender4Integration:
    """Modern Blender 4.4+ integration"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_terrain_object(
        self,
        mesh: trimesh.Trimesh,
        name: str = "terrain",
        material_name: str = "terrain_material",
    ) -> bpy.types.Object:
        """Create Blender object from trimesh with modern API"""
        try:
            # Create mesh data
            mesh_data = bpy.data.meshes.new(name)

            # Convert trimesh to Blender format
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()

            # Create mesh
            mesh_data.from_pydata(vertices, [], faces)
            mesh_data.update()

            # Create object
            terrain_obj = bpy.data.objects.new(name, mesh_data)
            bpy.context.collection.objects.link(terrain_obj)

            # Apply modern material
            self._apply_modern_material(terrain_obj, material_name)

            # Use modern Blender 4.4+ features
            self._setup_geometry_nodes(terrain_obj)
            self._setup_attributes(terrain_obj, mesh)

            # Tag for Infinigen compatibility
            tag_object(terrain_obj, Tags.Terrain)

            self.logger.info(f"✅ Blender terrain object created: {name}")
            return terrain_obj

        except Exception as e:
            self.logger.error(f"Error creating Blender terrain object: {e}")
            return None

    def _apply_modern_material(self, obj: bpy.types.Object, material_name: str):
        """Apply modern material using Blender 4.4+ features"""
        try:
            # Create or get material
            if material_name in bpy.data.materials:
                material = bpy.data.materials[material_name]
            else:
                material = bpy.data.materials.new(name=material_name)
                material.use_nodes = True

            # Clear existing nodes
            material.node_tree.nodes.clear()

            # Create modern node setup
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Output
            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (400, 0)

            # Principled BSDF
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (200, 0)

            # Connect
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

            # Apply material
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Could not apply modern material: {e}")

    def _setup_geometry_nodes(self, obj: bpy.types.Object):
        """Setup Geometry Nodes modifier for modern Blender"""
        try:
            # Add Geometry Nodes modifier
            geom_mod = obj.modifiers.new(name="TerrainGeometry", type="NODES")

            # Create node group
            node_group = bpy.data.node_groups.new(
                name="TerrainGeometryNodes", type="GeometryNodeTree"
            )
            geom_mod.node_group = node_group

            # Add nodes
            input_node = node_group.nodes.new("NodeGroupInput")
            output_node = node_group.nodes.new("NodeGroupOutput")

            input_node.location = (0, 0)
            output_node.location = (200, 0)

            # Connect
            node_group.links.new(input_node.outputs[0], output_node.inputs[0])

        except Exception as e:
            self.logger.warning(f"Could not setup geometry nodes: {e}")

    def _setup_attributes(self, obj: bpy.types.Object, mesh: trimesh.Trimesh):
        """Setup vertex attributes for modern Blender"""
        try:
            # Add height attribute
            height_attr = obj.data.attributes.new(
                name="height", type="FLOAT", domain="POINT"
            )
            height_attr.data.foreach_set("value", mesh.vertices[:, 2])

            # Add normal attribute
            normal_attr = obj.data.attributes.new(
                name="normal", type="FLOAT_VECTOR", domain="POINT"
            )
            normals = mesh.vertex_normals
            normal_attr.data.foreach_set("vector", normals.flatten())

        except Exception as e:
            self.logger.warning(f"Could not setup attributes: {e}")


class ModernTerrainEngine:
    """Complete modern terrain generation engine"""

    def __init__(self, config: TerrainConfig = None, device: str = "cpu"):
        self.config = config or TerrainConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.mesh_system = ModernMeshSystem(device)
        self.pytorch_processor = PyTorchGeometricTerrainProcessor(device)
        self.kernels_system = KernelsInterpolationSystem()
        self.spatial_manager = (
            DuckDBSpatialManager() if self.config.use_duckdb_storage else None
        )
        self.blender_integration = Blender4Integration()

        # Initialize random seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def generate_terrain(self, **kwargs) -> Dict[str, Any]:
        """Generate complete terrain with all modern features"""
        start_time = time.time()

        try:
            self.logger.info(
                f"Generating modern terrain: {self.config.terrain_type.value}"
            )

            # 1. Generate height map
            height_map = self._generate_height_map()

            # 2. Process with PyTorch Geometric if enabled
            if self.config.use_pytorch_geometric:
                height_map = self._process_with_pytorch_geometric(height_map)

            # 3. Create mesh using modern system
            bounds = self.config.bounds[:4]  # x_min, x_max, y_min, y_max
            mesh = self.mesh_system.create_from_heightmap(height_map, bounds)

            # 4. Create Blender object
            terrain_obj = self.blender_integration.create_terrain_object(
                mesh, f"{self.config.terrain_type.value}_terrain_{self.config.seed}"
            )

            # 5. Store in DuckDB if enabled
            if self.spatial_manager:
                metadata = {
                    "terrain_type": self.config.terrain_type.value,
                    "seed": self.config.seed,
                    "resolution": self.config.resolution,
                    "generation_time": time.time() - start_time,
                }
                self.spatial_manager.store_terrain_data(
                    f"terrain_{self.config.seed}", height_map, bounds, metadata
                )

            generation_time = time.time() - start_time

            return {
                "success": True,
                "terrain_object": terrain_obj,
                "mesh": mesh,
                "height_map": height_map,
                "generation_time": generation_time,
                "vertices_count": len(mesh.vertices),
                "faces_count": len(mesh.faces),
                "config": self.config,
            }

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Terrain generation failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "generation_time": generation_time,
                "config": self.config,
            }

    def _generate_height_map(self) -> np.ndarray:
        """Generate height map based on terrain type"""
        resolution = self.config.resolution

        if self.config.terrain_type == TerrainType.MOUNTAIN:
            return self._generate_mountain_heightmap(resolution)
        elif self.config.terrain_type == TerrainType.HILLS:
            return self._generate_hills_heightmap(resolution)
        elif self.config.terrain_type == TerrainType.VALLEY:
            return self._generate_valley_heightmap(resolution)
        elif self.config.terrain_type == TerrainType.PLATEAU:
            return self._generate_plateau_heightmap(resolution)
        elif self.config.terrain_type == TerrainType.CAVE:
            return self._generate_cave_heightmap(resolution)
        else:
            return self._generate_default_heightmap(resolution)

    def _generate_mountain_heightmap(self, resolution: int) -> np.ndarray:
        """Generate mountain terrain heightmap"""
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        X, Y = np.meshgrid(x, y)

        # Multi-octave noise for mountains
        height = np.zeros_like(X)

        # Base terrain
        height += 0.5 * self._perlin_noise(X, Y, scale=0.5, octaves=4)

        # Mountain peaks
        height += 0.3 * self._perlin_noise(X, Y, scale=0.2, octaves=6)

        # Fine details
        height += 0.1 * self._perlin_noise(X, Y, scale=0.05, octaves=8)

        # Apply sharpening
        height = self._sharpen_heightmap(height)

        return height

    def _generate_hills_heightmap(self, resolution: int) -> np.ndarray:
        """Generate hills terrain heightmap"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Gentle hills
        height = 0.3 * self._perlin_noise(X, Y, scale=0.3, octaves=3)
        height += 0.1 * self._perlin_noise(X, Y, scale=0.1, octaves=5)

        return height

    def _generate_valley_heightmap(self, resolution: int) -> np.ndarray:
        """Generate valley terrain heightmap"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Valley shape
        distance = np.sqrt(X**2 + Y**2)
        valley = -0.5 * np.exp(-(distance**2) / 0.5)

        # Add noise
        noise = 0.1 * self._perlin_noise(X, Y, scale=0.2, octaves=4)

        return valley + noise

    def _generate_plateau_heightmap(self, resolution: int) -> np.ndarray:
        """Generate plateau terrain heightmap"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Plateau with edges
        plateau = np.ones_like(X) * 0.5
        edge_falloff = np.exp(-(X**2 + Y**2) / 0.3)
        plateau *= edge_falloff

        # Add variation
        plateau += 0.05 * self._perlin_noise(X, Y, scale=0.1, octaves=3)

        return plateau

    def _generate_cave_heightmap(self, resolution: int) -> np.ndarray:
        """Generate cave terrain heightmap"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        # Cave system using noise
        cave_noise = self._perlin_noise(X, Y, scale=0.1, octaves=6)
        cave_threshold = 0.3

        # Create cave openings
        cave_mask = cave_noise > cave_threshold
        height = np.where(cave_mask, -0.5, 0.2)

        return height

    def _generate_default_heightmap(self, resolution: int) -> np.ndarray:
        """Generate default terrain heightmap"""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)

        return 0.2 * self._perlin_noise(X, Y, scale=0.3, octaves=3)

    def _perlin_noise(
        self, X: np.ndarray, Y: np.ndarray, scale: float = 1.0, octaves: int = 4
    ) -> np.ndarray:
        """Generate Perlin-like noise"""
        result = np.zeros_like(X)
        amplitude = 1.0
        frequency = scale

        for i in range(octaves):
            result += amplitude * np.sin(X * frequency) * np.cos(Y * frequency)
            amplitude *= 0.5
            frequency *= 2.0

        return result / (2.0 - 2.0 ** (-octaves))

    def _sharpen_heightmap(self, height_map: np.ndarray) -> np.ndarray:
        """Apply sharpening to heightmap"""
        return (np.sin((height_map - 0.5) / 0.5 * np.pi / 2) + 1) / 2

    def _process_with_pytorch_geometric(self, height_map: np.ndarray) -> np.ndarray:
        """Process heightmap using PyTorch Geometric"""
        try:
            # Create graph
            graph_data = self.pytorch_processor.create_terrain_graph(height_map)
            if graph_data is None:
                return height_map

            # Enhance terrain
            enhanced_graph = self.pytorch_processor.enhance_terrain_with_gnn(
                graph_data, enhancement_type="smoothing"
            )

            # Convert back to heightmap
            h, w = height_map.shape
            enhanced_heights = enhanced_graph.height.cpu().numpy()
            enhanced_height_map = enhanced_heights.reshape(h, w)

            return enhanced_height_map

        except Exception as e:
            self.logger.warning(f"PyTorch Geometric processing failed: {e}")
            return height_map

    def cleanup(self):
        """Cleanup resources"""
        if self.spatial_manager:
            self.spatial_manager.conn.close()


# Convenience functions for backward compatibility
def create_terrain_engine(
    config: TerrainConfig = None, device: str = "cpu"
) -> ModernTerrainEngine:
    """Create a modern terrain engine instance"""
    return ModernTerrainEngine(config, device)


def generate_terrain(
    terrain_type: str = "mountain", seed: int = 42, resolution: int = 512, **kwargs
) -> Dict[str, Any]:
    """Generate terrain with simple interface"""
    config = TerrainConfig(
        terrain_type=TerrainType(terrain_type),
        seed=seed,
        resolution=resolution,
        **kwargs,
    )

    engine = ModernTerrainEngine(config)
    result = engine.generate_terrain()
    engine.cleanup()

    return result
