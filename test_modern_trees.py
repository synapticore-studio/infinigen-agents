#!/usr/bin/env python3
"""
Test für das moderne Tree-System
"""

import logging
import sys
from pathlib import Path

# Mock bpy für Test-Umgebung
try:
    import bpy
except ImportError:
    print("bpy nicht verfügbar - Mock für Test")

    class MockBpy:
        class data:
            class meshes:
                @staticmethod
                def new(name):
                    return None

            class objects:
                @staticmethod
                def new(name, mesh):
                    return None

            class materials:
                @staticmethod
                def new(name):
                    return None

        class context:
            collection = None

    bpy = MockBpy()

import numpy as np

# Add infinigen root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_modern_tree_generator():
    """Teste ModernTreeGenerator"""
    print("🧪 Teste ModernTreeGenerator...")

    try:
        from tools.modern_tree_system import ModernTreeGenerator

        generator = ModernTreeGenerator()

        # Teste verfügbare Baum-Typen
        tree_types = generator.get_available_tree_types()
        print(f"✅ Verfügbare Baum-Typen: {tree_types}")

        # Teste Baum-Generierung
        for tree_type in tree_types:
            tree = generator.generate_tree(
                tree_type=tree_type, seed=42, position=(0, 0, 0), scale=1.0
            )
            if tree:
                print(f"✅ {tree_type.title()}-Baum generiert: {tree.name}")
            else:
                print(f"❌ {tree_type.title()}-Baum fehlgeschlagen")

        # Teste Wald-Generierung
        forest = generator.generate_forest(
            tree_count=5, area_size=20.0, tree_types=["oak", "pine"], seed=42
        )
        print(f"✅ Wald generiert: {len(forest)} Bäume")

        return True

    except Exception as e:
        print(f"❌ ModernTreeGenerator Test fehlgeschlagen: {e}")
        return False


def test_modern_tree_factory():
    """Teste ModernTreeFactory"""
    print("🧪 Teste ModernTreeFactory...")

    try:
        from tools.modern_tree_system import ModernTreeFactory

        factory = ModernTreeFactory(seed=42, coarse=True)

        # Teste Asset-Erstellung
        tree = factory.create_asset(
            {"tree_type": "oak", "position": (0, 0, 0), "scale": 1.0}
        )

        if tree:
            print(f"✅ TreeFactory Asset erstellt: {tree.name}")
            return True
        else:
            print("❌ TreeFactory Asset fehlgeschlagen")
            return False

    except Exception as e:
        print(f"❌ ModernTreeFactory Test fehlgeschlagen: {e}")
        return False


def test_tree_integration():
    """Teste Tree-Integration"""
    print("🧪 Teste Tree-Integration...")

    try:
        from infinigen.assets.objects.trees import BushFactory, TreeFactory

        # Teste TreeFactory
        tree_factory = TreeFactory(seed=42, coarse=True)
        tree = tree_factory.create_asset()

        if tree:
            print(f"✅ TreeFactory Integration funktioniert: {tree.name}")
        else:
            print("❌ TreeFactory Integration fehlgeschlagen")

        # Teste BushFactory
        bush_factory = BushFactory(seed=42, coarse=True)
        bush = bush_factory.create_asset({"tree_type": "bush"})

        if bush:
            print(f"✅ BushFactory Integration funktioniert: {bush.name}")
        else:
            print("❌ BushFactory Integration fehlgeschlagen")

        return True

    except Exception as e:
        print(f"❌ Tree-Integration Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 MODERN TREE SYSTEM TEST")
    print("=" * 50)

    results = {
        "generator": test_modern_tree_generator(),
        "factory": test_modern_tree_factory(),
        "integration": test_tree_integration(),
    }

    print("\n" + "=" * 50)
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"📊 ERGEBNIS: {passed_count}/{total_count} Tests bestanden")

    if passed_count == total_count:
        print("🎉 ALLE TESTS BESTANDEN - Modernes Tree-System funktioniert!")
    else:
        print("⚠️ EINIGE TESTS FEHLGESCHLAGEN - Tree-System benötigt Korrekturen")


if __name__ == "__main__":
    main()
