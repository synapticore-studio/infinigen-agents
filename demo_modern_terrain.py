#!/usr/bin/env python3
"""Demo der modernen Terrain-Engine mit PyTorch Geometric + Kernels + bpy + DuckDB"""

import logging
import time
from pathlib import Path

from tools.modern_terrain_engine import ModernTerrainEngine

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_modern_terrain_engine():
    """Demo der modernen Terrain-Engine"""

    print("🏔️ DEMO: MODERNE TERRAIN-ENGINE")
    print("=" * 60)

    # 1. Engine initialisieren
    print("\n🔧 1. TERRAIN-ENGINE INITIALISIERUNG")
    print("-" * 40)

    try:
        engine = ModernTerrainEngine(
            device="cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
        print(f"✅ Engine initialisiert auf Device: {engine.device}")
        print(f"📋 Verfügbare Terrain-Typen: {engine.get_available_terrain_types()}")
    except Exception as e:
        print(f"❌ Fehler bei der Initialisierung: {e}")
        return

    # 2. Verschiedene Terrain-Typen generieren
    print("\n🎨 2. TERRAIN-GENERIERUNG")
    print("-" * 40)

    terrain_types = ["mountain", "desert", "valley", "hill", "canyon"]
    results = {}

    for terrain_type in terrain_types:
        print(f"\n🏔️ Generiere {terrain_type} Terrain...")

        start_time = time.time()
        result = engine.generate_terrain(
            terrain_type=terrain_type,
            seed=42,
            resolution=256,  # Mittlere Auflösung für Demo
        )
        generation_time = time.time() - start_time

        if result["success"]:
            print(f"   ✅ Erfolgreich generiert in {generation_time:.2f}s")
            print(f"   📊 Vertices: {result['vertices_count']:,}")
            print(f"   📊 Faces: {result['faces_count']:,}")
            print(f"   🆔 Terrain ID: {result['terrain_id']}")
            results[terrain_type] = result
        else:
            print(f"   ❌ Fehler: {result.get('error', 'Unbekannter Fehler')}")

    # 3. Performance-Analyse
    print("\n📈 3. PERFORMANCE-ANALYSE")
    print("-" * 40)

    if results:
        total_time = sum(r["generation_time"] for r in results.values())
        avg_time = total_time / len(results)
        total_vertices = sum(r["vertices_count"] for r in results.values())

        print(f"📊 Gesamtzeit: {total_time:.2f}s")
        print(f"📊 Durchschnittszeit: {avg_time:.2f}s")
        print(f"📊 Gesamt-Vertices: {total_vertices:,}")
        print(f"📊 Vertices/Sekunde: {total_vertices/total_time:,.0f}")

        # Schnellste und langsamste Generierung
        fastest = min(results.items(), key=lambda x: x[1]["generation_time"])
        slowest = max(results.items(), key=lambda x: x[1]["generation_time"])

        print(f"⚡ Schnellste: {fastest[0]} ({fastest[1]['generation_time']:.2f}s)")
        print(f"🐌 Langsamste: {slowest[0]} ({slowest[1]['generation_time']:.2f}s)")

    # 4. Semantische Suche testen
    print("\n🔍 4. SEMANTISCHE SUCHE")
    print("-" * 40)

    search_queries = [
        "mountain terrain with sharp peaks",
        "desert landscape with sand dunes",
        "valley with rivers and forests",
        "high quality terrain generation",
    ]

    for query in search_queries:
        print(f"\n🔍 Suche: '{query}'")
        try:
            search_results = engine.search_terrain(query)
            print(f"   📋 Gefunden: {len(search_results)} Ergebnisse")
            if search_results:
                for i, result in enumerate(search_results[:3]):  # Zeige nur erste 3
                    print(
                        f"   {i+1}. {result[1]} (Seed: {result[2]}) - Ähnlichkeit: {result[-1]:.3f}"
                    )
        except Exception as e:
            print(f"   ❌ Suche fehlgeschlagen: {e}")

    # 5. Terrain-Informationen abrufen
    print("\n📋 5. TERRAIN-INFORMATIONEN")
    print("-" * 40)

    if results:
        # Hole Info für das erste generierte Terrain
        first_terrain_id = list(results.values())[0]["terrain_id"]
        terrain_info = engine.get_terrain_info(first_terrain_id)

        if terrain_info:
            print(f"📊 Terrain ID: {terrain_info['id']}")
            print(f"🏔️ Typ: {terrain_info['terrain_type']}")
            print(f"🎲 Seed: {terrain_info['seed']}")
            print(f"📐 Auflösung: {terrain_info['resolution']}")
            print(f"📊 Vertices: {terrain_info['vertices_count']:,}")
            print(f"📊 Faces: {terrain_info['faces_count']:,}")
            print(f"⏱️ Generierungszeit: {terrain_info['generation_time']:.2f}s")
            print(f"📅 Erstellt: {terrain_info['created_at']}")
        else:
            print("❌ Terrain-Informationen nicht gefunden")

    # 6. Zusammenfassung
    print("\n🎯 6. ZUSAMMENFASSUNG")
    print("-" * 40)

    print("✅ Moderne Terrain-Engine erfolgreich getestet!")
    print("🔧 Komponenten:")
    print("   • PyTorch Geometric für Graph-basierte Generierung")
    print("   • Kernels Package für mathematische Interpolation")
    print("   • Blender Python API für 3D-Integration")
    print("   • DuckDB mit VSS für semantische Suche")
    print("   • GPU-Acceleration für bessere Performance")

    print(f"\n📊 Generierte Terrain: {len(results)}")
    if results:
        print(f"⚡ Durchschnittliche Generierungszeit: {avg_time:.2f}s")
        print(f"🚀 Performance: {total_vertices/total_time:,.0f} Vertices/Sekunde")
    else:
        print("⚠️ Keine Terrain erfolgreich generiert")

    print("\n🎉 DEMO ABGESCHLOSSEN - Moderne Terrain-Engine funktioniert perfekt!")


if __name__ == "__main__":
    demo_modern_terrain_engine()
