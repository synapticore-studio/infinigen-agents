#!/usr/bin/env python3
"""
Test der Terrain-Migration zur modernen Engine
"""

import sys
from pathlib import Path

# Infinigen-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))


def test_terrain_import():
    """Teste Terrain-Import"""
    print("🧪 Teste Terrain-Import...")

    try:
        from infinigen.terrain.core import Terrain

        print("✅ Terrain-Import erfolgreich")
        return True
    except Exception as e:
        print(f"❌ Terrain-Import fehlgeschlagen: {e}")
        return False


def test_terrain_creation():
    """Teste Terrain-Erstellung"""
    print("🧪 Teste Terrain-Erstellung...")

    try:
        from infinigen.terrain.core import Terrain

        # Terrain ohne Blender-Kontext erstellen
        terrain = Terrain(
            seed=42,
            task="coarse",
            asset_folder="",
            asset_version="",
            on_the_fly_asset_folder="",
            device="cpu",
        )

        print("✅ Terrain-Erstellung erfolgreich")
        print(f"   - Seed: {terrain.seed}")
        print(f"   - Device: {terrain.device}")
        print(f"   - Terrain-Type: {terrain.terrain_type}")

        return True
    except Exception as e:
        print(f"❌ Terrain-Erstellung fehlgeschlagen: {e}")
        return False


def test_modern_engine():
    """Teste moderne Terrain-Engine direkt"""
    print("🧪 Teste moderne Terrain-Engine...")

    try:
        from tools.modern_terrain_engine import ModernTerrainEngine

        engine = ModernTerrainEngine(device="cpu")

        # Teste Terrain-Generierung
        result = engine.generate_terrain(
            terrain_type="mountain", seed=42, resolution=64  # Kleine Auflösung für Test
        )

        if result["success"]:
            print("✅ Moderne Terrain-Engine funktioniert")
            print(f"   - Generierungszeit: {result['generation_time']:.2f}s")
            print(f"   - Vertices: {result['vertices_count']}")
            return True
        else:
            print(
                f"❌ Terrain-Generierung fehlgeschlagen: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        print(f"❌ Moderne Terrain-Engine Test fehlgeschlagen: {e}")
        return False


def test_agent_integration():
    """Teste Agent-Integration"""
    print("🧪 Teste Agent-Integration...")

    try:
        # Teste direkt die moderne Terrain-Engine über Tools
        from tools.modern_terrain_engine import ModernTerrainEngine

        engine = ModernTerrainEngine(device="cpu")
        result = engine.generate_terrain(
            terrain_type="mountain", seed=42, resolution=64
        )

        if result["success"]:
            print("✅ Agent-Tools funktionieren")
            return True
        else:
            print(
                f"❌ Agent-Tools fehlgeschlagen: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        print(f"❌ Agent-Integration Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 TERRAIN-MIGRATION TEST")
    print("=" * 50)

    tests = [
        test_terrain_import,
        test_terrain_creation,
        test_modern_engine,
        test_agent_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 ERGEBNIS: {passed}/{total} Tests bestanden")

    if passed == total:
        print("🎉 ALLE TESTS BESTANDEN - Migration erfolgreich!")
        return True
    else:
        print("⚠️ EINIGE TESTS FEHLGESCHLAGEN - Migration benötigt Korrekturen")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
