#!/usr/bin/env python3
"""
Terrain System Test Runner
Run comprehensive integration tests for the modern terrain system
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main test runner"""
    print("🧪 Modern Terrain System - Integration Test Suite")
    print("=" * 60)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    try:
        # Import and run tests
        from infinigen.terrain.tests import run_integration_tests

        logger.info("Starting integration tests...")
        success = run_integration_tests()

        if success:
            print("\n✅ All tests passed!")
            logger.info("Integration tests completed successfully")
            return 0
        else:
            print("\n❌ Some tests failed!")
            logger.error("Integration tests failed")
            return 1

    except ImportError as e:
        logger.error(f"Could not import test modules: {e}")
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  uv add pytest")
        print("  uv sync")
        return 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ Test execution error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
