#!/usr/bin/env python3
"""
Test der vollständigen Terrain-Engine
"""

import sys
from pathlib import Path

# Infinigen-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))


def test_complete_terrain_engine():
    """Teste die vollständige Terrain-Engine"""
    print("🧪 Teste vollständige Terrain-Engine...")

    try:
        from tools.complete_terrain_engine import CompleteTerrainEngine

        # Engine initialisieren
        engine = CompleteTerrainEngine(device="cpu")

        # Verschiedene Terrain-Typen testen
        terrain_types = engine.get_available_terrain_types()
        print(f"   Verfügbare Terrain-Typen: {terrain_types}")

        results = []
        for terrain_type in terrain_types[:3]:  # Teste erste 3 Typen
            print(f"   Generiere {terrain_type}...")

            result = engine.generate_terrain(
                terrain_type=terrain_type,
                seed=42,
                resolution=32,  # Kleine Auflösung für schnellen Test
            )

            if result["success"]:
                print(f"   ✅ {terrain_type}: {result['generation_time']:.2f}s")
                results.append(True)

                # Prüfe Maps
                maps = [
                    "height_map",
                    "normal_map",
                    "displacement_map",
                    "roughness_map",
                    "ao_map",
                ]
                for map_name in maps:
                    if map_name in result and result[map_name] is not None:
                        print(f"      ✅ {map_name}: {result[map_name].shape}")
                    else:
                        print(f"      ❌ {map_name}: fehlt")
            else:
                print(f"   ❌ {terrain_type}: {result.get('error', 'Unknown error')}")
                results.append(False)

        # Cleanup
        engine.cleanup()

        success_count = sum(results)
        total_count = len(results)

        print(f"📊 {success_count}/{total_count} Terrain-Typen erfolgreich generiert")
        return success_count == total_count

    except Exception as e:
        print(f"❌ Vollständige Terrain-Engine Test fehlgeschlagen: {e}")
        return False


def test_terrain_adapter():
    """Teste Terrain-Adapter mit vollständiger Engine"""
    print("🧪 Teste Terrain-Adapter...")

    try:
        from infinigen.terrain.modern_adapter import ModernTerrainAdapter

        # Adapter erstellen
        terrain = ModernTerrainAdapter(seed=42, task="coarse", device="cpu")

        # Coarse Terrain generieren
        print("   Generiere Coarse Terrain...")
        coarse_mesh = terrain.coarse_terrain()

        if coarse_mesh:
            print(f"   ✅ Coarse Terrain: {coarse_mesh.name}")

            # Prüfe Materialien
            if coarse_mesh.data.materials:
                print(f"   ✅ Materialien: {len(coarse_mesh.data.materials)}")
            else:
                print("   ⚠️ Keine Materialien")

            return True
        else:
            print("   ❌ Coarse Terrain konnte nicht generiert werden")
            return False

    except Exception as e:
        print(f"❌ Terrain-Adapter Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 COMPLETE TERRAIN ENGINE TEST")
    print("=" * 50)

    tests = [
        test_complete_terrain_engine,
        test_terrain_adapter,
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
        print("🎉 ALLE COMPLETE TERRAIN TESTS BESTANDEN!")
        return True
    else:
        print("⚠️ EINIGE COMPLETE TERRAIN TESTS FEHLGESCHLAGEN")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
