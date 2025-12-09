#!/usr/bin/env python3
"""
Test der erweiterten Terrain-Engine mit allen Maps
"""

import sys
from pathlib import Path

# Infinigen-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))


def test_advanced_terrain_engine():
    """Teste die erweiterte Terrain-Engine"""
    print("🧪 Teste erweiterte Terrain-Engine...")

    try:
        from tools.advanced_terrain_engine import AdvancedTerrainEngine

        # Engine initialisieren
        engine = AdvancedTerrainEngine(device="cpu")

        # Verschiedene Terrain-Typen testen
        terrain_types = engine.get_available_terrain_types()
        print(f"   Verfügbare Terrain-Typen: {terrain_types}")

        results = []
        for terrain_type in terrain_types[:3]:  # Teste erste 3 Typen
            print(f"   Generiere {terrain_type}...")

            result = engine.generate_terrain(
                terrain_type=terrain_type,
                seed=42,
                resolution=64,  # Kleine Auflösung für schnellen Test
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
        print(f"❌ Erweiterte Terrain-Engine Test fehlgeschlagen: {e}")
        return False


def test_terrain_maps():
    """Teste Terrain-Map-Generierung"""
    print("🧪 Teste Terrain-Map-Generierung...")

    try:
        from tools.advanced_terrain_engine import TerrainMapGenerator

        generator = TerrainMapGenerator(device="cpu")

        # Teste verschiedene Map-Typen
        height_map = generator.generate_height_map("mountain", 42, 32)
        normal_map = generator.generate_normal_map(height_map)
        displacement_map = generator.generate_displacement_map(height_map)
        roughness_map = generator.generate_roughness_map(height_map)
        ao_map = generator.generate_ao_map(height_map)

        maps = {
            "height_map": height_map,
            "normal_map": normal_map,
            "displacement_map": displacement_map,
            "roughness_map": roughness_map,
            "ao_map": ao_map,
        }

        all_valid = True
        for map_name, map_data in maps.items():
            if map_data is not None and map_data.size > 0:
                print(f"   ✅ {map_name}: {map_data.shape}")
            else:
                print(f"   ❌ {map_name}: ungültig")
                all_valid = False

        return all_valid

    except Exception as e:
        print(f"❌ Terrain-Map-Generierung Test fehlgeschlagen: {e}")
        return False


def test_blender_integration():
    """Teste Blender-Integration"""
    print("🧪 Teste Blender-Integration...")

    try:
        from tools.advanced_terrain_engine import (
            BlenderTerrainIntegrator,
            TerrainMapGenerator,
        )

        # Maps generieren
        generator = TerrainMapGenerator(device="cpu")
        height_map = generator.generate_height_map("hills", 42, 32)
        normal_map = generator.generate_normal_map(height_map)
        displacement_map = generator.generate_displacement_map(height_map)
        roughness_map = generator.generate_roughness_map(height_map)
        ao_map = generator.generate_ao_map(height_map)

        # Blender-Integration
        integrator = BlenderTerrainIntegrator()

        # Mesh erstellen
        terrain_mesh = integrator.create_terrain_mesh(height_map, "test_terrain")

        if terrain_mesh:
            print(f"   ✅ Blender-Mesh erstellt: {terrain_mesh.name}")

            # Materialien anwenden
            integrator.apply_terrain_materials(
                terrain_mesh,
                "hills",
                height_map,
                normal_map,
                displacement_map,
                roughness_map,
                ao_map,
            )

            print("   ✅ Materialien mit Maps angewendet")
            return True
        else:
            print("   ❌ Blender-Mesh konnte nicht erstellt werden")
            return False

    except Exception as e:
        print(f"❌ Blender-Integration Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 ADVANCED TERRAIN ENGINE TEST")
    print("=" * 50)

    tests = [
        test_terrain_maps,
        test_blender_integration,
        test_advanced_terrain_engine,
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
        print("🎉 ALLE ADVANCED TERRAIN TESTS BESTANDEN!")
        return True
    else:
        print("⚠️ EINIGE ADVANCED TERRAIN TESTS FEHLGESCHLAGEN")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
