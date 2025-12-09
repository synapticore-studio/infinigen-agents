#!/usr/bin/env python3
"""
Test Core Terrain Components
Tests only the core terrain components without Blender dependencies
"""

import logging
import sys
from pathlib import Path

# Add infinigen to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_imports():
    """Test core terrain imports"""
    try:
        logger.info("🧪 Testing core terrain imports...")

        # Test basic imports
        import numpy as np
        import torch

        logger.info("✅ NumPy and PyTorch imported")

        # Test GPyTorch
        try:
            import gpytorch

            logger.info("✅ GPyTorch imported")
        except ImportError:
            logger.warning("⚠️ GPyTorch not available")

        # Test PyTorch Geometric
        try:
            import torch_geometric

            logger.info("✅ PyTorch Geometric imported")
        except ImportError:
            logger.warning("⚠️ PyTorch Geometric not available")

        # Test DuckDB
        try:
            import duckdb

            logger.info("✅ DuckDB imported")
        except ImportError:
            logger.warning("⚠️ DuckDB not available")

        return True

    except Exception as e:
        logger.error(f"❌ Core imports test failed: {e}")
        return False


def test_terrain_maps():
    """Test terrain map generation"""
    try:
        logger.info("🧪 Testing terrain map generation...")

        import numpy as np
        from scipy.ndimage import gaussian_filter

        # Create test height map
        height, width = 64, 64
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

        # Mountain terrain
        height_map = np.exp(-(x**2 + y**2) / 0.5) * 20
        height_map += np.random.rand(height, width) * 5

        logger.info(f"✅ Height map created: {height_map.shape}")
        logger.info(f"   - Range: {height_map.min():.2f} to {height_map.max():.2f}")

        # Generate normal map
        gx, gy = np.gradient(height_map)
        norm = np.sqrt(gx**2 + gy**2 + 1.0)
        normal_map = np.stack([-gx / norm, -gy / norm, 1.0 / norm], axis=-1)
        normal_map = ((normal_map + 1.0) / 2.0 * 255).astype(np.uint8)

        logger.info(f"✅ Normal map created: {normal_map.shape}")

        # Generate displacement map
        displacement_map = (height_map - height_map.min()) / (
            height_map.max() - height_map.min() + 1e-8
        )
        logger.info(f"✅ Displacement map created: {displacement_map.shape}")

        # Generate roughness map
        kernel = np.ones((3, 3)) / 9
        mean = gaussian_filter(height_map, sigma=1)
        variance = gaussian_filter(height_map**2, sigma=1) - mean**2
        roughness_map = (variance - variance.min()) / (
            variance.max() - variance.min() + 1e-8
        )

        logger.info(f"✅ Roughness map created: {roughness_map.shape}")

        # Generate AO map
        ao_map = 1.0 - (height_map - height_map.min()) / (
            height_map.max() - height_map.min() + 1e-8
        )
        logger.info(f"✅ AO map created: {ao_map.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Terrain maps test failed: {e}")
        return False


def test_gpytorch_kernels():
    """Test GPyTorch kernels"""
    try:
        logger.info("🧪 Testing GPyTorch kernels...")

        import numpy as np
        import torch

        # Test if GPyTorch is available
        try:
            import gpytorch
            from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
            from gpytorch.likelihoods import GaussianLikelihood
            from gpytorch.means import ConstantMean
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from gpytorch.models import ExactGP

            # Create test data
            train_x = torch.randn(50, 2)
            train_y = torch.randn(50)

            # Define simple GP model
            class SimpleGPModel(ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = ConstantMean()
                    self.covar_module = ScaleKernel(RBFKernel())

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # Initialize and test
            likelihood = GaussianLikelihood()
            model = SimpleGPModel(train_x, train_y, likelihood)

            # Test forward pass
            model.eval()
            likelihood.eval()

            test_x = torch.randn(10, 2)
            with torch.no_grad():
                output = model(test_x)
                mean = output.mean
                variance = output.variance

            logger.info(f"✅ GPyTorch kernels working")
            logger.info(f"   - Test predictions: {mean.shape}")
            logger.info(f"   - Variance: {variance.shape}")

            return True

        except ImportError:
            logger.warning("⚠️ GPyTorch not available, skipping test")
            return True

    except Exception as e:
        logger.error(f"❌ GPyTorch kernels test failed: {e}")
        return False


def test_pytorch_geometric():
    """Test PyTorch Geometric"""
    try:
        logger.info("🧪 Testing PyTorch Geometric...")

        import numpy as np
        import torch

        # Test if PyTorch Geometric is available
        try:
            import torch_geometric
            from torch_geometric.data import Data
            from torch_geometric.nn import GATConv, GCNConv, GraphSAGE

            # Create test graph data
            x = torch.randn(100, 1)  # Node features
            edge_index = torch.randint(0, 100, (2, 200))  # Random edges
            data = Data(x=x, edge_index=edge_index)

            # Test GNN models
            gcn = GCNConv(1, 64)
            gat = GATConv(1, 64, heads=4, dropout=0.1)
            graphsage = GraphSAGE(1, 64, num_layers=2)

            # Test forward passes
            with torch.no_grad():
                gcn_out = gcn(data.x, data.edge_index)
                gat_out = gat(data.x, data.edge_index)
                graphsage_out = graphsage(data.x, data.edge_index)

            logger.info(f"✅ PyTorch Geometric working")
            logger.info(f"   - GCN output: {gcn_out.shape}")
            logger.info(f"   - GAT output: {gat_out.shape}")
            logger.info(f"   - GraphSAGE output: {graphsage_out.shape}")

            return True

        except ImportError:
            logger.warning("⚠️ PyTorch Geometric not available, skipping test")
            return True

    except Exception as e:
        logger.error(f"❌ PyTorch Geometric test failed: {e}")
        return False


def test_duckdb():
    """Test DuckDB"""
    try:
        logger.info("🧪 Testing DuckDB...")

        # Test if DuckDB is available
        try:
            import duckdb

            # Create test database
            conn = duckdb.connect(":memory:")

            # Create test table
            conn.execute(
                """
                CREATE TABLE terrain_data (
                    id INTEGER,
                    x FLOAT,
                    y FLOAT,
                    height FLOAT
                )
            """
            )

            # Insert test data
            import numpy as np

            data = np.random.rand(100, 4)
            conn.executemany(
                "INSERT INTO terrain_data VALUES (?, ?, ?, ?)", data.tolist()
            )

            # Query data
            result = conn.execute("SELECT COUNT(*) FROM terrain_data").fetchone()
            logger.info(f"✅ DuckDB working")
            logger.info(f"   - Records inserted: {result[0]}")

            conn.close()
            return True

        except ImportError:
            logger.warning("⚠️ DuckDB not available, skipping test")
            return True

    except Exception as e:
        logger.error(f"❌ DuckDB test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🧪 Starting Core Terrain Component Tests")

    tests = [
        ("Core Imports", test_core_imports),
        ("Terrain Maps", test_terrain_maps),
        ("GPyTorch Kernels", test_gpytorch_kernels),
        ("PyTorch Geometric", test_pytorch_geometric),
        ("DuckDB", test_duckdb),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            success = test_func()
            results.append((test_name, success))

            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        logger.info("🎉 All core tests PASSED!")
        return True
    else:
        logger.error("💥 Some core tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
