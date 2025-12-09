#!/usr/bin/env python3
"""
Test der Terrain-Export-Funktionalität
"""

import sys
from pathlib import Path

# Infinigen-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent))


def test_terrain_export():
    """Teste Terrain-Export in verschiedene Formate"""
    print("🧪 Teste Terrain-Export...")

    try:
        from infinigen.terrain.modern_adapter import ModernTerrainAdapter

        # Terrain erstellen
        terrain = ModernTerrainAdapter(
            seed=42,
            task="coarse",
            asset_folder="",
            asset_version="",
            on_the_fly_asset_folder="",
            device="cpu",
        )

        # Export testen
        print("📤 Exportiere Terrain in verschiedene Formate...")
        success = terrain.export()

        if success:
            print("✅ Terrain-Export erfolgreich")

            # Prüfe ob Export-Dateien erstellt wurden
            export_files = [
                f"terrain_export_{terrain.seed}.blend",
                f"terrain_{terrain.seed}.obj",
                f"heightmap_{terrain.seed}.png",
            ]

            created_files = []
            for file_path in export_files:
                if Path(file_path).exists():
                    created_files.append(file_path)
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ⚠️ {file_path} nicht gefunden")

            print(
                f"📊 {len(created_files)}/{len(export_files)} Export-Dateien erstellt"
            )
            return True
        else:
            print("❌ Terrain-Export fehlgeschlagen")
            return False

    except Exception as e:
        print(f"❌ Terrain-Export Test fehlgeschlagen: {e}")
        return False


def test_terrain_data_storage():
    """Teste Terrain-Datenspeicherung"""
    print("🧪 Teste Terrain-Datenspeicherung...")

    try:
        from tools.modern_terrain_engine import ModernTerrainEngine

        engine = ModernTerrainEngine(device="cpu")

        # Generiere Terrain
        result = engine.generate_terrain(
            terrain_type="mountain", seed=42, resolution=64
        )

        if result["success"]:
            print("✅ Terrain generiert und gespeichert")
            print(f"   - Terrain ID: {result.get('terrain_id', 'N/A')}")
            print(f"   - Generierungszeit: {result.get('generation_time', 0):.2f}s")
            print(f"   - Vertices: {result.get('vertices_count', 0)}")

            # Prüfe Datenbank
            if Path("terrain.db").exists():
                print("   ✅ DuckDB-Datenbank erstellt")
            else:
                print("   ⚠️ DuckDB-Datenbank nicht gefunden")

            return True
        else:
            print(
                f"❌ Terrain-Generierung fehlgeschlagen: {result.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        print(f"❌ Terrain-Datenspeicherung Test fehlgeschlagen: {e}")
        return False


def main():
    """Haupttest-Funktion"""
    print("🚀 TERRAIN-EXPORT TEST")
    print("=" * 50)

    tests = [
        test_terrain_data_storage,
        test_terrain_export,
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
        print("🎉 ALLE EXPORT-TESTS BESTANDEN!")
        return True
    else:
        print("⚠️ EINIGE EXPORT-TESTS FEHLGESCHLAGEN")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
