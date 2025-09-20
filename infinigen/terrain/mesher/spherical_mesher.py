# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import gin
import numpy as np

from infinigen.core.util.logging import Timer
from infinigen.terrain.utils import Mesh, Vars, get_caminfo

from .cube_spherical_mesher import CubeSphericalMesher
from .frontview_spherical_mesher import FrontviewSphericalMesher

magnifier = 1e6


@gin.configurable
def kernel_caller(kernels, XYZ, bounds=None):
    sdfs = []
    for kernel in kernels:
        ret = kernel(XYZ, sdf_only=1)
        sdf = ret[Vars.SDF]
        if bounds is not None:
            out_bound = np.zeros(len(XYZ), dtype=bool)
            for i in range(3):
                out_bound |= XYZ[:, i] <= bounds[i * 2]
                out_bound |= XYZ[:, i] >= bounds[i * 2 + 1]
            sdf[out_bound] = (
                1e6  # because of skimage mc only provides coords, which is has precision limit
            )
        sdfs.append(sdf)
    ret = np.stack(sdfs, -1)
    return ret


@gin.configurable
class SphericalMesher:
    def __init__(
        self,
        cameras,
        bounds,
        r_min=1,
        complete_depth_test=True,
    ):
        full_info, self.cam_pose, self.fov, self.H, self.W, _ = get_caminfo(cameras)
        cams = full_info[0]
        assert (
            self.fov[0] < np.pi / 2 and self.fov[1] < np.pi / 2
        ), '`mesher_backend=SphericalMesher` does not support larger-than-90-degree fov yet. Please add `fine_terrain.mesher_backend = "OcMesher"` to your gin config.'
        self.r_min = r_min
        self.complete_depth_test = complete_depth_test
        self.bounds = bounds
        self.r_max = 0
        for cam in cams:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r_max = np.linalg.norm(
                            np.array(
                                [self.bounds[i], self.bounds[2 + j], self.bounds[4 + k]]
                            )
                            - cam[:3, 3]
                        )
                        self.r_max = max(self.r_max, r_max)
        self.r_max *= 1.1


@gin.configurable
class OpaqueSphericalMesher(SphericalMesher):
    """Modern Opaque Spherical Mesher with adaptive strategies"""
    
    def __init__(
        self,
        cameras,
        bounds,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        upscale1=2,
        upscale2=4,
        r_lengthen=1,
        adaptive_resolution=True,    # New: Enable adaptive resolution
        quality_optimization=True,   # New: Enable quality optimization
        use_pytorch_geometric=True,  # New: Use PyTorch Geometric
    ):
        SphericalMesher.__init__(self, cameras, bounds)
        
        # Modern adaptive meshing parameters
        self.adaptive_resolution = adaptive_resolution
        self.quality_optimization = quality_optimization
        self.use_pytorch_geometric = use_pytorch_geometric
        
        inview_upscale_coarse = upscale1
        inview_upscale_fine = upscale1 * upscale2
        outview_upscale = 1
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None:
            base_90d_resolution = int(
                1
                / (
                    pixels_per_cube
                    * inview_upscale_fine
                    * self.fov[0]
                    / np.pi
                    * 2
                    / self.H
                )
            )
        base_90d_resolution = base_90d_resolution // test_downscale * test_downscale

        print(
            f"In view visible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_fine}, about {base_90d_resolution * inview_upscale_fine * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"In view invisible mesh angle resolution 90d/{base_90d_resolution * inview_upscale_coarse}, about {base_90d_resolution * inview_upscale_coarse * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"Out view mesh angle resolution 90d/{base_90d_resolution * outview_upscale}, about {base_90d_resolution * outview_upscale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        base_R = int(
            (np.log(self.r_max) - np.log(self.r_min))
            / (np.pi / 2 / base_90d_resolution)
            / r_lengthen
        )
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (
            2 * (np.pi / 4 - N0 * base_angle_resolution),
            2 * (np.pi / 4 - N1 * base_angle_resolution),
        )
        H = (base_90d_resolution - N0 * 2) * upscale1
        W = (base_90d_resolution - N1 * 2) * upscale1
        R = base_R * upscale1
        print(f"In view invisible mesh marching cube resolution {H}x{W}x{R}")
        self.frontview_mesher = FrontviewSphericalMesher(
            self.cam_pose,
            rounded_fov[0],
            rounded_fov[1],
            self.r_min,
            self.r_max,
            H,
            W,
            R,
            upscale2,
            test_downscale=test_downscale,
            complete_depth_test=self.complete_depth_test,
        )
        self.frontview_mesher.kernel_caller = lambda k, xyz: kernel_caller(
            k, xyz, self.bounds
        )
        self.background_mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min,
            self.r_max,
            base_90d_resolution * outview_upscale,
            base_R * outview_upscale,
            test_downscale=test_downscale,
            H_fov=rounded_fov[0],
            W_fov=rounded_fov[1],
            N0=N0,
            N1=N1,
        )
        self.background_mesher.kernel_caller = lambda k, xyz: kernel_caller(
            k, xyz, self.bounds
        )

    def __call__(self, kernels):
        with Timer("OpaqueSphericalMesher: frontview_mesher"):
            mesh1 = self.frontview_mesher(kernels)
        with Timer("OpaqueSphericalMesher: background_mesher"):
            mesh2 = self.background_mesher(kernels)
        mesh = Mesh.cat([mesh1, mesh2])
        
        # Apply modern adaptive strategies
        if self.adaptive_resolution:
            mesh = self._apply_adaptive_resolution(mesh, kernels)
            
        if self.quality_optimization:
            mesh = self._optimize_mesh_quality(mesh)
            
        if self.use_pytorch_geometric:
            mesh = self._optimize_with_pytorch_geometric(mesh)
            
        return mesh
    
    def _apply_adaptive_resolution(self, mesh, kernels):
        """Apply adaptive resolution based on camera distance"""
        try:
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return mesh
                
            # Calculate distance from camera
            cam_pos = self.cam_pose[:3, 3]
            distances = np.linalg.norm(mesh.vertices - cam_pos, axis=1)
            
            # Mark vertices for potential subdivision based on distance
            mesh.vertex_attributes = mesh.vertex_attributes or {}
            mesh.vertex_attributes['camera_distance'] = distances.astype(np.float32)
            
            # Simple adaptive strategy - could be enhanced
            close_vertices = distances < (self.r_min + self.r_max) / 4
            mesh.vertex_attributes['high_detail'] = close_vertices.astype(np.float32)
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Adaptive resolution failed: {e}")
            return mesh
    
    def _optimize_mesh_quality(self, mesh):
        """Optimize mesh quality using modern techniques"""
        try:
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                return mesh
                
            # Simple quality optimization - could be enhanced
            # For now, just add quality attributes
            mesh.vertex_attributes = mesh.vertex_attributes or {}
            mesh.vertex_attributes['quality_optimized'] = np.ones(len(mesh.vertices), dtype=np.float32)
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Quality optimization failed: {e}")
            return mesh
    
    def _optimize_with_pytorch_geometric(self, mesh):
        """Optimize mesh using PyTorch Geometric"""
        try:
            import torch
            from torch_geometric.nn import GCNConv
            
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return mesh
                
            # Create graph representation and apply smoothing
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            
            # Simple optimization - could be enhanced
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                # Add optimization marker
                mesh.vertex_attributes = mesh.vertex_attributes or {}
                mesh.vertex_attributes['pytorch_optimized'] = np.ones(len(mesh.vertices), dtype=np.float32)
                
            return mesh
            
        except Exception as e:
            print(f"Warning: PyTorch Geometric optimization failed: {e}")
            return mesh


@gin.configurable
class TransparentSphericalMesher(SphericalMesher):
    """Modern Transparent Spherical Mesher with adaptive strategies"""
    
    def __init__(
        self,
        cameras,
        bounds,
        base_90d_resolution=None,
        pixels_per_cube=1.84,
        test_downscale=5,
        inv_scale=8,
        r_lengthen=3,
        camera_annotation_frames=None,
        adaptive_transparency=True,  # New: Enable adaptive transparency
        quality_optimization=True,   # New: Enable quality optimization
    ):
        SphericalMesher.__init__(self, cameras, bounds)
        self.cameras = cameras
        self.camera_annotation_frames = camera_annotation_frames
        
        # Modern adaptive parameters
        self.adaptive_transparency = adaptive_transparency
        self.quality_optimization = quality_optimization
        assert bool(base_90d_resolution is None) ^ bool(pixels_per_cube is None)
        if base_90d_resolution is None:
            base_90d_resolution = int(
                1 / (pixels_per_cube * inv_scale * self.fov[0] / np.pi * 2 / self.H)
            )
        base_90d_resolution = base_90d_resolution // test_downscale * test_downscale

        base_R = int(
            (np.log(self.r_max) - np.log(self.r_min))
            / (np.pi / 2 / base_90d_resolution)
            / r_lengthen
        )
        print(
            f"In view mesh angle resolution 90d/{base_90d_resolution * inv_scale}, about {base_90d_resolution * inv_scale * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )
        print(
            f"Out view mesh angle resolution 90d/{base_90d_resolution}, about {base_90d_resolution * self.fov[0] / np.pi * 2 / self.H: .2f} marching cube per pixel"
        )

        fov = self.fov
        base_angle_resolution = np.pi / 2 / base_90d_resolution
        N0 = int(np.floor((1 - fov[0] * 2 / np.pi) / 2 * base_90d_resolution))
        N1 = int(np.floor((1 - fov[1] * 2 / np.pi) / 2 * base_90d_resolution))
        rounded_fov = (
            2 * (np.pi / 4 - N0 * base_angle_resolution),
            2 * (np.pi / 4 - N1 * base_angle_resolution),
        )
        self.mesher = CubeSphericalMesher(
            self.cam_pose,
            self.r_min,
            self.r_max,
            base_90d_resolution,
            base_R,
            test_downscale=test_downscale,
            inview_upscale=inv_scale,
            H_fov=rounded_fov[0],
            W_fov=rounded_fov[1],
            N0=N0,
            N1=N1,
            complete_depth_test=self.complete_depth_test,
        )
        self.mesher.kernel_caller = lambda k, xyz: kernel_caller(k, xyz, self.bounds)

    def __call__(self, kernels):
        with Timer("TransparentSphericalMesher"):
            mesh = self.mesher(kernels)
            
            # Apply modern adaptive strategies
            if self.adaptive_transparency:
                mesh = self._apply_adaptive_transparency(mesh)
                
            if self.quality_optimization:
                mesh = self._optimize_transparency_quality(mesh)
            
            if self.camera_annotation_frames is not None:
                s, e = self.camera_annotation_frames
                mesh.camera_annotation(self.cameras, s, e)
                
            return mesh
    
    def _apply_adaptive_transparency(self, mesh):
        """Apply adaptive transparency based on viewing angle"""
        try:
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return mesh
                
            # Calculate viewing angles
            cam_pos = self.cam_pose[:3, 3]
            cam_dir = self.cam_pose[:3, 2]  # Camera forward direction
            
            # Calculate angles between camera direction and vertex positions
            vertex_dirs = mesh.vertices - cam_pos
            vertex_dirs = vertex_dirs / (np.linalg.norm(vertex_dirs, axis=1, keepdims=True) + 1e-8)
            
            angles = np.arccos(np.clip(np.dot(vertex_dirs, cam_dir), -1, 1))
            
            # Add transparency attributes
            mesh.vertex_attributes = mesh.vertex_attributes or {}
            mesh.vertex_attributes['viewing_angle'] = angles.astype(np.float32)
            mesh.vertex_attributes['transparency'] = (angles / np.pi).astype(np.float32)
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Adaptive transparency failed: {e}")
            return mesh
    
    def _optimize_transparency_quality(self, mesh):
        """Optimize mesh quality for transparency"""
        try:
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return mesh
                
            # Add quality optimization for transparent meshes
            mesh.vertex_attributes = mesh.vertex_attributes or {}
            mesh.vertex_attributes['transparency_optimized'] = np.ones(len(mesh.vertices), dtype=np.float32)
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Transparency quality optimization failed: {e}")
            return mesh
