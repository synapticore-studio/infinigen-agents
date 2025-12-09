# Astronomy Extension für Infinigen

## Übersicht

Das Astronomy Extension erweitert Infinigen um realistische astronomische Objekte und Szenen. Es nutzt Infinigen's Core-System für Export, Rendering und Workflow-Management.

## Verfügbare Objekte

### Planeten
- **RockyPlanetFactory**: Gesteinsplaneten (Erde, Mars)
- **GasGiantFactory**: Gasriesen (Jupiter, Saturn) mit optionaler Atmosphäre
- **IcePlanetFactory**: Eisplaneten (Uranus, Neptune)
- **RingedPlanetFactory**: Planeten mit Ringen (Saturn)

### Monde
- **MoonFactory**: Monde mit realistischen Orbital-Mechaniken
- Unterstützt Kepler-Gesetze und Roche-Limits
- Automatische Animation von Umlaufbahnen

### Sterne & Nebel
- **StarFieldFactory**: Sternenfelder für Hintergründe
- **EmissionNebulaFactory**: Emissionsnebel
- **ReflectionNebulaFactory**: Reflexionsnebel
- **DarkNebulaFactory**: Dunkelnebel

### Asteroiden & Kometen
- **AsteroidBeltFactory**: Asteroidengürtel
- **CometFactory**: Kometen mit Schweifen
- **MeteorFactory**: Meteore

### Raumstationen
- **SpaceStationFactory**: Raumstationen mit kinematischen Gelenken
- **SatelliteFactory**: Satelliten mit Solarpanels
- **SpacecraftFactory**: Raumschiffe

## Verwendung

### 1. Einfache Objekte erstellen

```python
from infinigen.assets.objects.astronomy.planets import RockyPlanetFactory

# Einfacher Planet
planet = RockyPlanetFactory(factory_seed=1, radius=1.0, resolution=64).create_asset()
planet.name = "My_Planet"
planet.location = (0, 0, 0)
```

### 2. Komplette Szenen

```python
from infinigen_examples.astronomy_workflow import AstronomyWorkflow

# Kompletter Workflow
workflow = AstronomyWorkflow(output_folder="my_astronomy_scene", scene_seed=42)
workflow.run_full_workflow()
```

### 3. Schnelle Objekte

```python
from infinigen_examples.quick_astronomy_objects import quick_planet, create_solar_system

# Einzelner Planet
planet = quick_planet("rocky", radius=1.0, location=(0, 0, 0))

# Komplettes Sonnensystem
solar_system = create_solar_system()
```

## Output-Formate

### Blender-Dateien
- `.blend`: Haupt-Blender-Dateien
- `.blend1`: Backup-Dateien

### USD-Export
- `.usdc`: Universal Scene Description (komprimiert)
- `.usda`: Universal Scene Description (ASCII)
- Texturen in separatem `textures/` Ordner

### Mesh-Export
- `.npz`: Komprimierte Mesh-Daten (NumPy-Format)
- `.json`: Mesh-Metadaten

### Texturen
- `*_DIFFUSE.png`: Diffuse/Albedo-Maps
- `*_NORMAL.png`: Normal-Maps
- `*_ROUGHNESS.png`: Roughness-Maps
- `*_METAL.png`: Metallic-Maps

## Beispiel-Skripte

### 1. `generate_astronomy.py`
Basis-Generierung von astronomischen Objekten

### 2. `create_astronomy_film.py`
Erstellt animierte Filme von astronomischen Szenen

### 3. `generate_3d_astronomy.py`
Generiert 3D-Objekte mit Infinigen's Export-System

### 4. `astronomy_workflow.py`
Vollständiger Workflow mit Rendering und Export

### 5. `quick_astronomy_objects.py`
Schnelle Generierung einzelner Objekte

## Konfiguration

### Gin-Konfiguration
```gin
# planets.gin
RockyPlanetFactory.radius = 1.0
RockyPlanetFactory.resolution = 64

GasGiantFactory.has_atmosphere = True
GasGiantFactory.has_rings = False

# constraints.gin
AstronomicalConstraintSolver.max_iterations = 200
AstronomicalConstraintSolver.tolerance = 0.005
```

### Parameter-Anpassung
```python
# Planet mit Atmosphäre
gas_giant = GasGiantFactory(
    factory_seed=1,
    radius=2.0,
    resolution=64,
    has_atmosphere=True
).create_asset()

# Mond mit realistischen Orbits
moon = MoonFactory(
    factory_seed=2,
    radius=0.3,
    orbit_radius=3.0,
    parent_planet=planet,
    use_realistic_orbits=True
).create_asset()
```

## Core-Integration

Das Astronomy Extension nutzt vollständig Infinigen's Core-System:

- **Constraints**: Orbital-Mechaniken und physikalische Gesetze
- **SimObjects**: Kinematische Strukturen für Simulationen
- **Export-System**: USD, NPZ, JSON-Formate
- **Tagging**: Objekt-Kategorisierung und -Verfolgung
- **Gin-Config**: Flexible Parameter-Anpassung

## Output-Locations

```
astronomy_output/
├── astronomy_scene.blend          # Haupt-Blender-Datei
├── export_astronomy_scene.blend/  # USD-Export
│   ├── export_astronomy_scene.usdc
│   └── textures/                  # Alle Texturen
├── frame_0001/                    # Mesh-Export
│   └── static_mesh/
│       ├── saved_mesh_0001.npz
│       └── saved_mesh.json
└── astronomy_render.png           # Render-Output
```

## Erweiterte Features

### Realistische Physik
- Kepler'sche Gesetze für Orbital-Mechaniken
- Roche-Limits für Stabilität
- Tidal Locking-Berechnungen

### Animation
- Automatische Umlaufbahn-Animationen
- Rotations-Animationen
- Kometen-Schweif-Animationen

### Materialien
- Procedural generierte Oberflächen
- Realistische Atmosphären-Effekte
- Ring-Systeme mit Partikeln

### Constraints
- Orbital-Constraints für stabile Orbits
- Planetary-System-Constraints für ganze Systeme
- Constraint-Solver für automatische Optimierung

## Troubleshooting

### Häufige Probleme

1. **Import-Fehler**: Stelle sicher, dass alle Dependencies installiert sind
2. **Export-Fehler**: Prüfe, ob Output-Verzeichnis existiert und beschreibbar ist
3. **Rendering-Fehler**: Überprüfe Blender-Version und Cycles-Engine

### Debug-Modus

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Dann deine Astronomy-Code ausführen
```

## Weiterentwicklung

Das Astronomy Extension ist modular aufgebaut und kann einfach erweitert werden:

- Neue Objekt-Typen in `infinigen/assets/objects/astronomy/`
- Neue Constraints in `infinigen/assets/objects/astronomy/constraints.py`
- Neue Materialien in den jeweiligen Factory-Klassen
- Neue Animationen in den Factory-Klassen

## Lizenz

Teil von Infinigen - Princeton University
