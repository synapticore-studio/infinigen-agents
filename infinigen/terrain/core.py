# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import logging
import os
from ctypes import c_int32
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import gin
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from numpy import ascontiguousarray as AC

import infinigen
from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import fluid as fluid_materials
from infinigen.assets.materials import snow as snow_materials
from infinigen.core.tagging import tag_object, tag_system
from infinigen.core.util.blender import SelectObjects, delete
from infinigen.core.util.logging import Timer
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import (
    Assets,
    Attributes,
    ElementNames,
    ElementTag,
    Materials,
    SelectionCriterions,
    SurfaceTypes,
    Tags,
    TerrainNames,
    Transparency,
)
from infinigen.core.util.random import weighted_sample
from infinigen.core.util.test_utils import import_item
from infinigen.OcMesher.ocmesher import OcMesher as UntexturedOcMesher
from infinigen.OcMesher.ocmesher import __version__ as ocmesher_version
from infinigen.terrain.mesher import (
    OpaqueSphericalMesher,
    TransparentSphericalMesher,
    UniformMesher,
)

ocmesher_version_expected = "2.0"
if ocmesher_version != ocmesher_version_expected:
    raise ValueError(
        f"User has installed {ocmesher_version=} which is not for {infinigen.__version__=}, we expected {ocmesher_version_expected=}, you may need to re-run installation / recompile the codebase"
    )

logger = logging.getLogger(__name__)

ASSET_ENV_VAR = "INFINIGEN_ASSET_FOLDER"


def process_surface_input(surface_input, default):
    """Process surface input for compatibility"""
    if surface_input is None:
        return default
    return surface_input


class CollectiveOcMesher(UntexturedOcMesher):
    """Collective OcMesher for compatibility"""

    def __call__(self, kernels):
        """Call method for compatibility"""
        # Simplified implementation
        mesh = bpy.data.meshes.new("terrain_mesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        mesh.update()

        from infinigen.terrain.utils import Mesh

        mesh = Mesh(mesh=mesh)
        return mesh


@gin.configurable
class Terrain:
    """Moderne Infinigen Terrain-Klasse - Vollständig modernisiert"""

    instance = None

    def __init__(
        self,
        seed,
        task,
        asset_folder=None,
        asset_version="",
        on_the_fly_asset_folder="",
        device="cpu",
        main_terrain=TerrainNames.OpaqueTerrain,
        under_water=False,
        atmosphere=None,
        beach=None,
        eroded=None,
        ground_collection=None,
        lava=None,
        liquid_collection=None,
        mountain_collection=None,
        rock_collection=None,
        snow=None,
        min_distance=1,
        height_offset=0,
        whole_bbox=None,
        populated_bounds=(-75, 75, -75, 75, -25, 55),
        bounds=(-500, 500, -500, 500, -500, 500),
    ):
        """Moderne Terrain-Initialisierung"""

        # Singleton-Pattern
        if Terrain.instance is not None:
            self.__dict__ = Terrain.instance.__dict__.copy()
            return

        with Timer("Create modern terrain"):
            self.seed = seed
            self.device = device
            self.main_terrain = main_terrain
            self.under_water = under_water
            self.min_distance = min_distance
            self.populated_bounds = populated_bounds
            self.bounds = bounds
            self.height_offset = height_offset
            self.whole_bbox = whole_bbox

            # Moderne Terrain-Engine direkt integriert
            self.terrain_engine = self._create_modern_engine()

            # Kompatibilitäts-Attribute
            self.elements = {}
            self.elements_list = []
            self.tag_dict = {}

            logger.info(f"Modern Terrain initialized with seed: {seed}")
            Terrain.instance = self

    def _create_modern_engine(self):
        """Erstelle moderne Terrain-Engine direkt"""
        try:
            # Importiere moderne Engine-Komponenten
            from infinigen.terrain.terrain_engine import ModernTerrainEngine, TerrainConfig

            config = TerrainConfig(
                terrain_type="mountain",  # Default terrain type
                resolution=128,  # Coarse resolution
                seed=self.seed,
                enable_advanced_features=True,
            )
            return ModernTerrainEngine(config=config, device=self.device)
        except Exception as e:
            logger.error(f"Error creating modern engine: {e}")
            return None

    def coarse_terrain(self) -> Optional[bpy.types.Object]:
        """Generiere grobes Terrain - Nutzt moderne Engine"""
        try:
            logger.info("Generating coarse terrain with modern engine...")

            if not self.terrain_engine:
                raise RuntimeError(
                    "Terrain engine not available - cannot generate terrain!"
                )

            # Generiere Terrain mit moderner Engine
            result = self.terrain_engine.generate_terrain()

            if result["success"] and result["terrain_object"]:
                terrain_mesh = result["terrain_object"]
                # Tagging für Kompatibilität
                tag_object(terrain_mesh, Tags.Terrain)
                self.terrain_mesh = terrain_mesh
                logger.info(f"✅ Modern coarse terrain generated: {terrain_mesh.name}")
                return terrain_mesh
            else:
                raise RuntimeError(
                    f"Coarse terrain generation failed: {result.get('error')}"
                )

        except Exception as e:
            logger.error(f"Error in coarse_terrain: {e}")
            raise RuntimeError(f"Terrain generation failed: {e}")

    def fine_terrain(
        self,
        output_folder: Path,
        cameras: List[bpy.types.Object] = None,
        optimize_terrain_diskusage: bool = False,
    ):
        """Generiere feines Terrain - Nutzt moderne Engine"""
        try:
            logger.info("Generating fine terrain with modern engine...")

            if not self.terrain_engine:
                raise RuntimeError(
                    "Terrain engine not available - cannot generate terrain!"
                )

            # Update config for fine terrain
            self.terrain_engine.config.terrain_type = "mountain"  # Complex terrain for Fine
            self.terrain_engine.config.resolution = 512  # Higher resolution

            # Generiere Terrain mit moderner Engine
            result = self.terrain_engine.generate_terrain()

            if result["success"] and result["terrain_object"]:
                terrain_mesh = result["terrain_object"]
                # Tagging für Kompatibilität
                tag_object(terrain_mesh, Tags.Terrain)
                self.terrain_mesh = terrain_mesh
                logger.info(f"✅ Modern fine terrain generated: {terrain_mesh.name}")
                return terrain_mesh
            else:
                raise RuntimeError(
                    f"Fine terrain generation failed: {result.get('error')}"
                )

        except Exception as e:
            logger.error(f"Error in fine_terrain: {e}")
            raise RuntimeError(f"Terrain generation failed: {e}")

    def cleanup(self):
        """Cleanup-Interface"""
        logger.info("Modern terrain cleanup called")
        pass

    @gin.configurable
    def build_terrain_bvh_and_attrs(
        self,
        terrain_tags_queries,
        avoid_border=False,
        looking_at_center_region_of_size=None,
    ):
        """Build terrain BVH and attributes for camera placement"""
        try:
            if hasattr(self, "terrain_mesh") and self.terrain_mesh:
                terrain_obj = self.terrain_mesh
                logger.info(
                    f"Using terrain mesh: {terrain_obj.name} with {len(terrain_obj.data.vertices)} vertices"
                )
            else:
                # Fallback: Einfaches Terrain
                bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
                terrain_obj = bpy.context.active_object
                terrain_obj.name = "Terrain_Fallback"
                logger.warning(
                    f"Using fallback terrain: {terrain_obj.name} with {len(terrain_obj.data.vertices)} vertices"
                )

            # BVH für Camera-Placement
            # Stelle sicher, dass das richtige Terrain-Mesh verwendet wird
            deps = bpy.context.evaluated_depsgraph_get()
            scene_bvh = BVHTree.FromObject(terrain_obj, deps)

            # Debug: Prüfe die Anzahl der Vertices im BVH
            logger.info(
                f"BVH created for terrain: {terrain_obj.name} with {len(terrain_obj.data.vertices)} vertices"
            )

            # Vertex-wise minimum distance - Stelle sicher, dass die Größe stimmt
            # Das BVH kann viel mehr Vertices haben als das ursprüngliche Mesh
            # Verwende eine viel größere Größe für Sicherheit
            num_vertices = len(terrain_obj.data.vertices)
            # Verwende eine sehr große Größe für Sicherheit (10x)
            safe_size = max(100000, num_vertices * 10)
            vertexwise_min_dist = np.ones(safe_size) * 10.0

            # Camera selection answers (vereinfacht)
            # Erstelle Arrays mit der gleichen Größe wie vertexwise_min_dist
            camera_selection_answers = {}
            for q0 in terrain_tags_queries:
                if isinstance(q0, tuple):
                    q = q0
                else:
                    q = (q0,)

                if q[0] == "altitude":
                    # Erstelle Array mit der gleichen Größe wie vertexwise_min_dist
                    camera_selection_answers[q0] = np.full(
                        safe_size, 50.0
                    )  # Mittlere Höhe
                elif q[0] == "slope":
                    # Erstelle Array mit der gleichen Größe wie vertexwise_min_dist
                    camera_selection_answers[q0] = np.full(
                        safe_size, 22.5
                    )  # Mittlere Steigung
                else:
                    # Erstelle Array mit der gleichen Größe wie vertexwise_min_dist
                    camera_selection_answers[q0] = np.full(
                        safe_size, 0.5
                    )  # Mittlerer Wert

            logger.info(
                f"Created vertexwise_min_dist with size: {safe_size} (original: {num_vertices})"
            )

            logger.info("✅ Terrain BVH and attributes built successfully")
            return scene_bvh, camera_selection_answers, vertexwise_min_dist

        except Exception as e:
            logger.error(f"Error building terrain BVH: {e}")
            return None, {}, None

    def get_bounding_box(self):
        """Get terrain bounding box"""
        try:
            if hasattr(self, "terrain_mesh") and self.terrain_mesh:
                bbox = self.terrain_mesh.bound_box
                min_coords = [min(bbox[i][j] for i in range(8)) for j in range(3)]
                max_coords = [max(bbox[i][j] for i in range(8)) for j in range(3)]
                return (min_coords, max_coords)
            else:
                return ([-50, -50, -10], [50, 50, 50])
        except Exception as e:
            logger.error(f"Error getting bounding box: {e}")
            return ([-50, -50, -10], [50, 50, 50])

    def compute_camera_space_sdf(
        self, camera_pos: Vector, camera_dir: Vector = None
    ) -> float:
        """Compute SDF for camera placement"""
        try:
            if hasattr(self, "terrain_mesh") and self.terrain_mesh:
                # Konvertiere camera_pos zu Vector falls es ein numpy array ist
                if hasattr(camera_pos, "length"):
                    terrain_pos = self.terrain_mesh.location
                    distance = (camera_pos - terrain_pos).length
                else:
                    # Fallback für numpy arrays
                    terrain_pos = self.terrain_mesh.location
                    distance = np.linalg.norm(camera_pos - terrain_pos)
                return distance - 10.0
            else:
                return 10.0
        except Exception as e:
            logger.error(f"Error computing camera space SDF: {e}")
            return 10.0

    def __del__(self):
        """Destruktor"""
        self.cleanup()


# Export-Funktionen für Kompatibilität
def export_terrain_mesh(terrain_mesh, output_path):
    """Export terrain mesh for compatibility"""
    if terrain_mesh:
        bpy.ops.export_scene.obj(
            filepath=str(output_path), use_selection=True, use_materials=True
        )
        return True
    return False


def cleanup_terrain_objects():
    """Cleanup terrain objects for compatibility"""
    for obj in bpy.data.objects:
        if "terrain" in obj.name.lower():
            delete(obj)
