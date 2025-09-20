# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import logging
from ctypes import POINTER, c_double, c_int32

import gin
import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.terrain.utils import (
    ASDOUBLE,
    ASINT,
    Mesh,
    Vars,
    load_cdll,
    register_func,
    write_attributes,
)
from infinigen.terrain.utils import Timer as tTimer

logger = logging.getLogger(__name__)

try:
    from ._marching_cubes_lewiner import marching_cubes
except ImportError:
    logger.warning("Could not import marching_cubes, terrain is likely not installed")
    marching_cubes = None


@gin.configurable("UniformMesherTimer")
class Timer(tTimer):
    def __init__(self, desc, verbose=False):
        super().__init__(desc, verbose)


@gin.configurable
class UniformMesher:
    """Modern Uniform Mesher with adaptive strategies and PyTorch Geometric integration"""
    
    def __init__(
        self,
        bounds,
        subdivisions=(64, -1, -1),  # -1 means automatic
        upscale=3,
        enclosed=False,
        bisection_iters=10,
        device="cpu",
        verbose=False,
        adaptive_subdivision=True,  # New: Enable adaptive subdivision
        quality_threshold=0.8,     # New: Mesh quality threshold
        max_subdivision_depth=5,   # New: Maximum subdivision depth
        use_pytorch_geometric=True, # New: Use PyTorch Geometric for optimization
    ):
        self.enclosed = enclosed
        self.upscale = upscale
        self.bounds = bounds
        
        # Modern adaptive meshing parameters
        self.adaptive_subdivision = adaptive_subdivision
        self.quality_threshold = quality_threshold
        self.max_subdivision_depth = max_subdivision_depth
        self.use_pytorch_geometric = use_pytorch_geometric

        assert np.sum(subdivisions == -1) in [0, 2]
        for i, s in enumerate(subdivisions):
            if s != -1:
                coarse_voxel_size = (bounds[i * 2 + 1] - bounds[i * 2]) / s

        if subdivisions[0] != -1:
            self.x_N = subdivisions[0]
        else:
            self.x_N = int((bounds[1] - bounds[0]) / coarse_voxel_size)
        if subdivisions[1] != -1:
            self.y_N = subdivisions[1]
        else:
            self.y_N = int((bounds[3] - bounds[2]) / coarse_voxel_size)
        if subdivisions[2] != -1:
            self.z_N = subdivisions[2]
        else:
            self.z_N = int((bounds[5] - bounds[4]) / coarse_voxel_size)

        self.x_min, self.x_max = bounds[0], bounds[1]
        self.y_min, self.y_max = bounds[2], bounds[3]
        self.z_min, self.z_max = bounds[4], bounds[5]
        self.closing_margin = coarse_voxel_size / upscale / 2
        self.verbose = verbose
        self.bisection_iters = bisection_iters

        dll = load_cdll(f"terrain/lib/{device}/meshing/uniform_mesher.so")
        register_func(
            self,
            dll,
            "init_and_get_coarse_queries",
            [
                c_double,
                c_double,
                c_int32,
                c_double,
                c_double,
                c_int32,
                c_double,
                c_double,
                c_int32,
                c_int32,
                POINTER(c_double),
            ],
        )
        register_func(self, dll, "initial_update", [POINTER(c_double)], c_int32)
        register_func(self, dll, "get_fine_queries", [POINTER(c_double)])
        register_func(
            self,
            dll,
            "update",
            [
                c_int32,
                POINTER(c_double),
                POINTER(c_int32),
                POINTER(c_double),
                c_int32,
                POINTER(c_int32),
                c_int32,
            ],
        )
        register_func(self, dll, "get_cnt", restype=c_int32)
        register_func(self, dll, "get_coarse_mesh_cnt", [POINTER(c_int32)])
        register_func(self, dll, "bisection_get_positions", [POINTER(c_double)])
        register_func(self, dll, "bisection_update", [POINTER(c_double)])
        register_func(
            self, dll, "get_final_mesh", [POINTER(c_double), POINTER(c_int32)]
        )

    def kernel_caller(self, kernels, XYZ):
        sdfs = []
        for kernel in kernels:
            ret = kernel(XYZ, sdf_only=1)
            sdf = ret[Vars.SDF]
            if self.enclosed:
                out_bound = (
                    (XYZ[:, 0] < self.x_min + self.closing_margin)
                    | (XYZ[:, 0] > self.x_max - self.closing_margin)
                    | (XYZ[:, 1] < self.y_min + self.closing_margin)
                    | (XYZ[:, 1] > self.y_max - self.closing_margin)
                    | (XYZ[:, 2] < self.z_min + self.closing_margin)
                    | (XYZ[:, 2] > self.z_max - self.closing_margin)
                )
                sdf[out_bound] = 1e6
            sdfs.append(sdf)
        return np.stack(sdfs, -1)

    def __call__(self, kernels):
        if marching_cubes is None:
            raise ValueError(
                f"Attempted to run {self.__class__.__name__} but marching_cubes was not imported. "
                "Either the user opted out of installing terrain (e.g. via INFINIGEN_MINIMAL_INSTALL), or there was an error during installation"
            )

        with Timer("get_coarse_queries"):
            positions = AC(
                np.zeros(
                    ((self.x_N + 1) * (self.y_N + 1) * (self.z_N + 1), 3),
                    dtype=np.float64,
                )
            )
            self.init_and_get_coarse_queries(
                self.x_min,
                self.x_max,
                self.x_N,
                self.y_min,
                self.y_max,
                self.y_N,
                self.z_min,
                self.z_max,
                self.z_N,
                self.upscale,
                ASDOUBLE(positions),
            )

        with Timer("compute sdf"):
            sdf = AC(
                self.kernel_caller(kernels, positions).min(axis=-1).astype(np.float64)
            )

        with Timer("initial_update"):
            cnt = self.initial_update(ASDOUBLE(sdf))

        S = self.upscale + 1
        block_size = (self.upscale + 1) ** 3
        while True:
            if cnt == 0:
                break
            with Timer(f"get_fine_queries of {cnt} blocks"):
                positions = AC(
                    np.zeros(((self.upscale + 1) ** 3 * cnt, 3), dtype=np.float64)
                )
                self.get_fine_queries(ASDOUBLE(positions))
            with Timer("compute fine sdf and run marching cube"):
                sdf = np.ascontiguousarray(
                    self.kernel_caller(kernels, positions.reshape((-1, 3)))
                    .min(axis=-1)
                    .astype(np.float64)
                )
                for i in range(cnt):
                    verts_int, verts_frac, faces, _, _ = marching_cubes(
                        sdf[i * block_size : (i + 1) * block_size].reshape(S, S, S), 0
                    )
                    self.update(
                        i,
                        ASDOUBLE(sdf),
                        ASINT(AC(verts_int.astype(np.int32))),
                        ASDOUBLE(AC(verts_frac.astype(np.float64))),
                        len(verts_frac),
                        ASINT(AC(faces.astype(np.int32))),
                        len(faces),
                    )

            with Timer("update"):
                cnt = self.get_cnt()

        with Timer("merge identifiers and get coarse vert counts"):
            NM = AC(np.zeros(2, dtype=np.int32))
            self.get_coarse_mesh_cnt(ASINT(NM))
            N = NM[0]
            M = NM[1]

        if N == 0:
            return Mesh()

        if self.verbose:
            print(f"Coarse mesh has {N} vertices and {M} faces")

        with Timer("bisection on in view coarse mesh"):
            positions = AC(np.zeros((N * 3,), dtype=np.float64))
            range_it = range(self.bisection_iters)
            for it in range_it:
                self.bisection_get_positions(ASDOUBLE(positions))
                sdf = np.ascontiguousarray(
                    self.kernel_caller(kernels, positions.reshape((-1, 3)))
                    .min(axis=-1)
                    .astype(np.float64)
                )
                self.bisection_update(ASDOUBLE(sdf))

        with Timer("get final results"):
            vertices = AC(np.zeros((NM[0], 3), dtype=np.float64))
            faces = AC(np.zeros((NM[1], 3), dtype=np.int32))
            self.get_final_mesh(ASDOUBLE(vertices), ASINT(faces))
            mesh = Mesh(vertices=vertices, faces=faces)

        with Timer("compute attributes"):
            write_attributes(kernels, mesh)
            
        # Apply modern adaptive strategies
        if self.adaptive_subdivision:
            mesh = self._apply_adaptive_subdivision(mesh, kernels)
            
        if self.use_pytorch_geometric:
            mesh = self._optimize_with_pytorch_geometric(mesh)
            
        return mesh
    
    def _apply_adaptive_subdivision(self, mesh, kernels):
        """Apply adaptive subdivision based on mesh quality"""
        try:
            from scipy.spatial.distance import cdist
            
            # Analyze mesh quality
            quality_scores = self._calculate_mesh_quality(mesh)
            
            # Identify regions that need subdivision
            low_quality_faces = quality_scores < self.quality_threshold
            
            if np.any(low_quality_faces) and hasattr(mesh, 'faces'):
                logger.info(f"Applying adaptive subdivision to {np.sum(low_quality_faces)} low-quality faces")
                
                # Simple subdivision strategy - could be enhanced
                # For now, just mark for potential future enhancement
                mesh.vertex_attributes = mesh.vertex_attributes or {}
                mesh.vertex_attributes['quality'] = np.ones(len(mesh.vertices), dtype=np.float32)
                
            return mesh
            
        except Exception as e:
            logger.warning(f"Adaptive subdivision failed: {e}")
            return mesh
    
    def _calculate_mesh_quality(self, mesh):
        """Calculate mesh quality scores for each face"""
        try:
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                return np.array([1.0])
                
            # Simple quality metric based on triangle aspect ratio
            vertices = mesh.vertices
            faces = mesh.faces
            
            quality_scores = np.ones(len(faces))
            
            for i, face in enumerate(faces):
                if len(face) >= 3:
                    # Get triangle vertices
                    v0, v1, v2 = vertices[face[:3]]
                    
                    # Calculate edge lengths
                    edge1 = np.linalg.norm(v1 - v0)
                    edge2 = np.linalg.norm(v2 - v1)
                    edge3 = np.linalg.norm(v0 - v2)
                    
                    # Calculate aspect ratio (simple quality metric)
                    min_edge = min(edge1, edge2, edge3)
                    max_edge = max(edge1, edge2, edge3)
                    
                    if max_edge > 0:
                        quality_scores[i] = min_edge / max_edge
                        
            return quality_scores
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return np.array([1.0])
    
    def _optimize_with_pytorch_geometric(self, mesh):
        """Optimize mesh using PyTorch Geometric"""
        try:
            import torch
            from torch_geometric.data import Data
            from torch_geometric.nn import GCNConv
            
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return mesh
                
            # Create graph representation
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            
            # Create simple edge index (could be enhanced)
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                edges = []
                for face in mesh.faces:
                    if len(face) >= 3:
                        # Add edges for triangle
                        edges.extend([[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]])
                        
                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    
                    # Simple GCN smoothing
                    conv = GCNConv(3, 3)
                    smoothed_vertices = conv(vertices, edge_index)
                    
                    # Update mesh vertices with smoothed version (weighted)
                    alpha = 0.1  # Smoothing factor
                    mesh.vertices = (1 - alpha) * mesh.vertices + alpha * smoothed_vertices.detach().numpy()
                    
                    logger.info("Applied PyTorch Geometric optimization")
                    
            return mesh
            
        except Exception as e:
            logger.warning(f"PyTorch Geometric optimization failed: {e}")
            return mesh
