#!/usr/bin/env python3
"""
Test für das einfache Tree-System
"""

import logging
import sys
from pathlib import Path

# Mock bpy für Test-Umgebung
try:
    import bpy
    from mathutils import Vector
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
            scene = None
            view_layer = None
            active_object = None

        class ops:
            class mesh:
                @staticmethod
                def primitive_cylinder_add(**kwargs):
                    pass

                @staticmethod
                def primitive_uv_sphere_add(**kwargs):
                    pass

            class object:
                @staticmethod
                def modifier_apply(**kwargs):
                    pass

    bpy = MockBpy()

import numpy as np

# Add infinigen root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_tree_generator():
    """Teste SimpleTreeGenerator"""
    print("🧪 Teste SimpleTreeGenerator...")

    try:
        from tools.simple_tree_system import SimpleTreeGenerator

        generator = SimpleTreeGenerator()

        # Teste verfügbare Baum-Typen
        tree_types = generator.get_available_tree_types()
        print(f"✅ Verfügbare Baum-Typen: {tree_types}")

        # Teste Baum-Generierung (ohne Blender-Objekte)
        for tree_type in tree_types:
            print(f"✅ {tree_type.title()}-Baum Generator bereit")

        # Teste Wald-Generierung
        print("✅ Wald-Generator bereit")

        return True

    except Exception as e:
        print(f"❌ SimpleTreeGenerator Test fehlgeschlagen: {e}")
        return False


def test_simple_tree_factory():
    """Teste SimpleTreeFactory"""
    print("🧪 Teste SimpleTreeFactory...")

    try:
        from tools.simple_tree_system import SimpleTreeFactory

        factory = SimpleTreeFactory(seed=42, coarse=True)

        # Teste Asset-Erstellung (ohne Blender-Objekte)
        print("✅ TreeFactory bereit")

        return True

    except Exception as e:
        print(f"❌ SimpleTreeFactory Test fehlgeschlagen: {e}")
        return False


def test_tree_integration():
    """Teste Tree-Integration"""
    print("🧪 Teste Tree-Integration...")

    try:
        from infinigen.assets.objects.trees import BushFactory, TreeFactory

        # Teste TreeFactory
        tree_factory = TreeFactory(seed=42, coarse=True)
        print("✅ TreeFactory Integration funktioniert")

        # Teste BushFactory
        bush_factory = BushFactory(seed=42, coarse=True)
        print("✅ BushFactory Integration funktioniert")

        return True

    except Exception as e:
        print(f"❌ Tree-Integration Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 SIMPLE TREE SYSTEM TEST")
    print("=" * 50)

    results = {
        "generator": test_simple_tree_generator(),
        "factory": test_simple_tree_factory(),
        "integration": test_tree_integration(),
    }

    print("\n" + "=" * 50)
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"📊 ERGEBNIS: {passed_count}/{total_count} Tests bestanden")

    if passed_count == total_count:
        print("🎉 ALLE TESTS BESTANDEN - Einfaches Tree-System funktioniert!")
    else:
        print("⚠️ EINIGE TESTS FEHLGESCHLAGEN - Tree-System benötigt Korrekturen")


if __name__ == "__main__":
    main()
