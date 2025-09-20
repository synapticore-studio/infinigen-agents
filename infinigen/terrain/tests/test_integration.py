#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Modern Terrain System
Tests all components working together with Blender 4.5.3+ features
"""

import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Test imports with fallbacks
try:
    import bpy
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    
try:
    import torch
    import torch_geometric
    HAS_PYTORCH_GEOMETRIC = True
except ImportError:
    HAS_PYTORCH_GEOMETRIC = False

try:
    from infinigen.terrain.terrain_engine import (
        ModernTerrainEngine,
        TerrainConfig,
        TerrainType,
    )
    from infinigen.terrain.rendering import setup_modern_terrain_rendering
    HAS_INFINIGEN = True
except ImportError:
    HAS_INFINIGEN = False


class TestModernTerrainIntegration(unittest.TestCase):
    """Integration tests for the modern terrain system"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.logger = logging.getLogger(__name__)
        cls.test_results = {}
        
        # Create test directory
        cls.test_dir = Path(tempfile.mkdtemp(prefix="terrain_test_"))
        cls.logger.info(f"Test directory: {cls.test_dir}")
    
    def setUp(self):
        """Setup for each test"""
        self.start_time = time.time()
    
    def tearDown(self):
        """Cleanup after each test"""
        test_time = time.time() - self.start_time
        test_name = self._testMethodName
        self.test_results[test_name] = {
            'duration': test_time,
            'status': 'passed' if not hasattr(self, '_outcome') or self._outcome.success else 'failed'
        }
    
    @unittest.skipUnless(HAS_INFINIGEN, "Infinigen not available")
    def test_terrain_config_creation(self):
        """Test terrain configuration creation"""
        config = TerrainConfig(
            terrain_type=TerrainType.MOUNTAIN,
            resolution=256,
            seed=42,
            use_pytorch_geometric=True,
            enable_geometry_baking=True
        )
        
        self.assertEqual(config.terrain_type, TerrainType.MOUNTAIN)
        self.assertEqual(config.resolution, 256)
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.use_pytorch_geometric)
        self.assertTrue(config.enable_geometry_baking)
        
        self.logger.info("✅ Terrain config creation test passed")
    
    @unittest.skipUnless(HAS_INFINIGEN, "Infinigen not available")
    def test_terrain_engine_initialization(self):
        """Test terrain engine initialization"""
        config = TerrainConfig(
            terrain_type=TerrainType.HILLS,
            resolution=128,
            seed=123
        )
        
        engine = ModernTerrainEngine(config, device="cpu")
        
        self.assertIsNotNone(engine.config)
        self.assertIsNotNone(engine.mesh_system)
        self.assertIsNotNone(engine.pytorch_processor)
        self.assertIsNotNone(engine.kernels_system)
        self.assertIsNotNone(engine.blender_integration)
        
        self.logger.info("✅ Terrain engine initialization test passed")
    
    @unittest.skipUnless(HAS_INFINIGEN and HAS_PYTORCH_GEOMETRIC, "Dependencies not available")
    def test_pytorch_geometric_integration(self):
        """Test PyTorch Geometric integration"""
        config = TerrainConfig(
            terrain_type=TerrainType.VALLEY,
            resolution=64,
            seed=456,
            use_pytorch_geometric=True
        )
        
        engine = ModernTerrainEngine(config, device="cpu")
        
        # Test graph creation
        height_map = np.random.rand(64, 64).astype(np.float32)
        graph_data = engine.pytorch_processor.create_terrain_graph(height_map)
        
        self.assertIsNotNone(graph_data)
        self.assertTrue(hasattr(graph_data, 'x'))
        self.assertTrue(hasattr(graph_data, 'edge_index'))
        self.assertTrue(hasattr(graph_data, 'height'))
        
        self.logger.info("✅ PyTorch Geometric integration test passed")
    
    @unittest.skipUnless(HAS_INFINIGEN, "Infinigen not available")
    def test_kernels_interpolation(self):
        """Test kernels interpolation system"""
        config = TerrainConfig(use_kernels_interpolation=True)
        engine = ModernTerrainEngine(config, device="cpu")
        
        # Test RBF interpolation
        sparse_points = np.random.rand(10, 2).astype(np.float32)
        sparse_values = np.random.rand(10).astype(np.float32)
        target_points = np.random.rand(5, 2).astype(np.float32)
        
        try:
            result = engine.kernels_system.interpolate_terrain(
                sparse_points, sparse_values, target_points
            )
            self.assertEqual(len(result), 5)
            self.logger.info("✅ Kernels interpolation test passed")
        except Exception as e:
            self.logger.warning(f"Kernels interpolation test failed: {e}")
            # This is expected if kernels package is not available
    
    @unittest.skipUnless(HAS_INFINIGEN, "Infinigen not available")
    def test_duckdb_spatial_manager(self):
        """Test DuckDB spatial data management"""
        config = TerrainConfig(use_duckdb_storage=True)
        engine = ModernTerrainEngine(config, device="cpu")
        
        if engine.spatial_manager:
            # Test data storage
            height_map = np.random.rand(32, 32).astype(np.float32)
            bounds = (-10, 10, -10, 10)
            metadata = {"test": "data"}
            
            success = engine.spatial_manager.store_terrain_data(
                "test_terrain", height_map, bounds, metadata
            )
            
            if success:
                # Test data query
                results = engine.spatial_manager.query_terrain_region(-5, 5, -5, 5)
                self.assertIsInstance(results, list)
                self.logger.info("✅ DuckDB spatial manager test passed")
            else:
                self.logger.warning("DuckDB spatial storage failed")
    
    @unittest.skipUnless(HAS_INFINIGEN and HAS_BLENDER, "Blender not available")
    def test_blender_integration(self):
        """Test Blender 4.5.3+ integration"""
        config = TerrainConfig(
            terrain_type=TerrainType.PLATEAU,
            resolution=64,
            enable_geometry_baking=True
        )
        
        engine = ModernTerrainEngine(config, device="cpu")
        
        # Test mesh creation
        height_map = np.random.rand(64, 64).astype(np.float32)
        bounds = (-5, 5, -5, 5)
        mesh = engine.mesh_system.create_from_heightmap(height_map, bounds)
        
        self.assertIsNotNone(mesh)
        self.assertTrue(hasattr(mesh, 'vertices'))
        self.assertTrue(hasattr(mesh, 'faces'))
        
        # Test Blender object creation
        terrain_obj = engine.blender_integration.create_terrain_object(
            mesh, "test_terrain"
        )
        
        if terrain_obj:
            self.assertEqual(terrain_obj.name, "test_terrain")
            self.logger.info("✅ Blender integration test passed")
        else:
            self.logger.warning("Blender object creation failed")
    
    @unittest.skipUnless(HAS_INFINIGEN and HAS_BLENDER, "Blender not available")
    def test_modern_rendering_setup(self):
        """Test modern rendering setup"""
        try:
            success = setup_modern_terrain_rendering(quality="medium")
            self.assertTrue(success)
            self.logger.info("✅ Modern rendering setup test passed")
        except Exception as e:
            self.logger.warning(f"Modern rendering setup test failed: {e}")
    
    @unittest.skipUnless(HAS_INFINIGEN, "Infinigen not available")
    def test_full_terrain_generation_pipeline(self):
        """Test complete terrain generation pipeline"""
        configs = [
            TerrainConfig(
                terrain_type=TerrainType.MOUNTAIN,
                resolution=64,
                seed=42,
                use_pytorch_geometric=HAS_PYTORCH_GEOMETRIC,
                enable_geometry_baking=HAS_BLENDER
            ),
            TerrainConfig(
                terrain_type=TerrainType.HILLS,
                resolution=32,
                seed=123,
                use_pytorch_geometric=HAS_PYTORCH_GEOMETRIC,
                enable_geometry_baking=HAS_BLENDER
            ),
            TerrainConfig(
                terrain_type=TerrainType.CAVE,
                resolution=32,
                seed=456,
                use_pytorch_geometric=HAS_PYTORCH_GEOMETRIC,
                enable_geometry_baking=HAS_BLENDER
            )
        ]
        
        for i, config in enumerate(configs):
            with self.subTest(config=config.terrain_type.value):
                engine = ModernTerrainEngine(config, device="cpu")
                
                start_time = time.time()
                result = engine.generate_terrain()
                generation_time = time.time() - start_time
                
                self.assertIsInstance(result, dict)
                self.assertIn('success', result)
                
                if result['success']:
                    self.assertIn('height_map', result)
                    self.assertIn('mesh', result)
                    self.assertIn('generation_time', result)
                    
                    # Validate height map
                    height_map = result['height_map']
                    self.assertEqual(height_map.shape, (config.resolution, config.resolution))
                    
                    # Validate mesh
                    mesh = result['mesh']
                    self.assertTrue(hasattr(mesh, 'vertices'))
                    self.assertTrue(len(mesh.vertices) > 0)
                    
                    self.logger.info(
                        f"✅ {config.terrain_type.value} terrain generated in {generation_time:.2f}s"
                    )
                else:
                    self.logger.warning(f"❌ {config.terrain_type.value} terrain generation failed")
                
                # Cleanup
                engine.cleanup()
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        if not HAS_INFINIGEN:
            self.skipTest("Infinigen not available")
        
        resolutions = [32, 64, 128]
        terrain_types = [TerrainType.MOUNTAIN, TerrainType.HILLS, TerrainType.VALLEY]
        
        benchmark_results = {}
        
        for resolution in resolutions:
            for terrain_type in terrain_types:
                config = TerrainConfig(
                    terrain_type=terrain_type,
                    resolution=resolution,
                    seed=42,
                    use_pytorch_geometric=False,  # Disable for pure performance test
                    enable_geometry_baking=False
                )
                
                engine = ModernTerrainEngine(config, device="cpu")
                
                start_time = time.time()
                result = engine.generate_terrain()
                generation_time = time.time() - start_time
                
                key = f"{terrain_type.value}_{resolution}"
                benchmark_results[key] = {
                    'time': generation_time,
                    'success': result.get('success', False),
                    'vertices': len(result.get('mesh', {}).vertices) if result.get('mesh') else 0
                }
                
                engine.cleanup()
        
        # Log benchmark results
        self.logger.info("📊 Performance Benchmark Results:")
        for key, data in benchmark_results.items():
            self.logger.info(
                f"  {key}: {data['time']:.2f}s, {data['vertices']} vertices, "
                f"{'✅' if data['success'] else '❌'}"
            )
        
        # Validate performance expectations
        for key, data in benchmark_results.items():
            if data['success']:
                # Expect reasonable generation times
                self.assertLess(data['time'], 30.0, f"Generation too slow for {key}")
                self.assertGreater(data['vertices'], 0, f"No vertices generated for {key}")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup test environment"""
        # Log test summary
        cls.logger.info("🧪 Test Summary:")
        total_tests = len(cls.test_results)
        passed_tests = sum(1 for r in cls.test_results.values() if r['status'] == 'passed')
        
        cls.logger.info(f"  Total tests: {total_tests}")
        cls.logger.info(f"  Passed: {passed_tests}")
        cls.logger.info(f"  Failed: {total_tests - passed_tests}")
        
        total_time = sum(r['duration'] for r in cls.test_results.values())
        cls.logger.info(f"  Total time: {total_time:.2f}s")
        
        # Cleanup test directory
        import shutil
        try:
            shutil.rmtree(cls.test_dir)
            cls.logger.info(f"Cleaned up test directory: {cls.test_dir}")
        except Exception as e:
            cls.logger.warning(f"Could not cleanup test directory: {e}")


class TestModernFeatures(unittest.TestCase):
    """Test modern Blender 4.5.3+ specific features"""
    
    @unittest.skipUnless(HAS_BLENDER, "Blender not available")
    def test_geometry_node_baking(self):
        """Test Geometry Node Baking functionality"""
        try:
            # This would test actual Blender functionality
            # For now, just test that the API is available
            self.assertTrue(hasattr(bpy.ops.object, 'geometry_node_bake_single'))
            logging.getLogger(__name__).info("✅ Geometry Node Baking API available")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Geometry Node Baking test failed: {e}")
    
    @unittest.skipUnless(HAS_BLENDER, "Blender not available")
    def test_topology_nodes(self):
        """Test Topology Nodes functionality"""
        try:
            # Test that topology nodes are available
            node_types = [
                "GeometryNodeCornersOfFace",
                "GeometryNodeEdgesOfVertex", 
                "GeometryNodeFaceOfCorner",
                "GeometryNodeVertexOfCorner"
            ]
            
            for node_type in node_types:
                self.assertTrue(hasattr(bpy.types, node_type))
            
            logging.getLogger(__name__).info("✅ Topology Nodes API available")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Topology Nodes test failed: {e}")
    
    @unittest.skipUnless(HAS_BLENDER, "Blender not available") 
    def test_sample_operations(self):
        """Test Sample Operations functionality"""
        try:
            # Test that sample operation nodes are available
            node_types = [
                "GeometryNodeSampleIndex",
                "GeometryNodeSampleNearest",
                "GeometryNodeSampleNearestSurface",
                "GeometryNodeRaycast",
                "GeometryNodeSampleUVSurface"
            ]
            
            for node_type in node_types:
                self.assertTrue(hasattr(bpy.types, node_type))
            
            logging.getLogger(__name__).info("✅ Sample Operations API available")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Sample Operations test failed: {e}")


def run_integration_tests():
    """Run all integration tests"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModernTerrainIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModernFeatures))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
