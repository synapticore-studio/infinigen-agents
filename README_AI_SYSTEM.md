# Infinigen AI System

Ein spezialisiertes AI-Agent-System für Infinigen, das auf dem "Weniger ist mehr" Prinzip basiert. Jeder Agent hat minimale Dependencies und fokussierte Tools für maximale Effizienz.

## 🏗️ **Architektur**

### **Agent-Spezialisierung**
- **SceneComposerAgent**: Szenen-Komposition (Nature/Indoor)
- **AssetGeneratorAgent**: Asset-Generierung (Creatures, Trees, Materials)
- **TerrainEngineerAgent**: Terrain-Generierung und -Optimierung
- **RenderControllerAgent**: Rendering und Ground Truth
- **DataManagerAgent**: Job-Management und Daten-Pipeline
- **ExportSpecialistAgent**: Export und Format-Konvertierung

### **Dependency-Organisation**
```
deps/
├── core_deps.py      # Minimale Core-Dependencies
├── blender_deps.py   # Blender-spezifische Dependencies
├── terrain_deps.py   # Terrain-Generierung Dependencies
├── render_deps.py    # Rendering Dependencies
├── data_deps.py      # Data-Management Dependencies
└── export_deps.py    # Export Dependencies
```

### **Tool-Spezialisierung**
```
tools/
├── scene_tools.py    # Szenen-Komposition Tools
├── asset_tools.py    # Asset-Generierung Tools
├── terrain_tools.py  # Terrain Tools
├── render_tools.py   # Rendering Tools
├── data_tools.py     # Data-Management Tools
└── export_tools.py   # Export Tools
```

## 🚀 **Verwendung**

### **Basis-Setup**
```python
from infinigen_ai_system import InfinigenAISystem

# System initialisieren
system = InfinigenAISystem()

# Status prüfen
status = system.get_system_status()
print(status)
```

### **Komplette Szene erstellen**
```python
# Komplette Szene mit allen Features
result = system.create_complete_scene(
    output_folder=Path("./output/scene_001"),
    scene_seed=42,
    scene_type="forest",
    complexity="medium",
    include_terrain=True,
    include_rendering=True,
    include_export=True
)
```

### **Asset-Bibliothek generieren**
```python
# Asset-Bibliothek erstellen
result = system.generate_asset_library(
    output_folder=Path("./assets/library"),
    asset_types=["pine", "oak", "carnivore", "herbivore"],
    count_per_type=10,
    complexity="medium"
)
```

### **Data-Pipeline erstellen**
```python
# Data-Generierung Pipeline
scene_configs = [
    {"seed": 42, "type": "forest"},
    {"seed": 123, "type": "desert"},
    {"seed": 456, "type": "mountain"}
]

result = system.create_data_generation_pipeline(
    job_name="nature_dataset",
    output_folder=Path("./datasets/nature"),
    scene_configs=scene_configs,
    tasks=["coarse", "populate", "render", "ground_truth"]
)
```

## 🎯 **Agent-Details**

### **SceneComposerAgent**
**Dependencies:** Core + Blender
**Tools:** Scene-Komposition, Validierung
**Verwendung:**
```python
agent = system.scene_composer

# Nature-Szene
result = agent.compose_nature_scene(
    output_folder=Path("./scenes/forest"),
    scene_seed=42,
    scene_type="forest"
)

# Indoor-Szene
result = agent.compose_indoor_scene(
    output_folder=Path("./scenes/kitchen"),
    scene_seed=123,
    room_types=["kitchen", "living_room"]
)
```

### **AssetGeneratorAgent**
**Dependencies:** Blender
**Tools:** Asset-Generierung, Parameter-Management
**Verwendung:**
```python
agent = system.asset_generator

# Creature generieren
result = agent.generate_creature_asset(
    creature_type="carnivore",
    output_path=Path("./assets/creature"),
    seed=42,
    complexity="high"
)

# Tree generieren
result = agent.generate_tree_asset(
    tree_type="pine",
    output_path=Path("./assets/tree"),
    seed=123,
    complexity="medium"
)
```

### **TerrainEngineerAgent**
**Dependencies:** Terrain + Core
**Tools:** Terrain-Generierung, Optimierung
**Verwendung:**
```python
agent = system.terrain_engineer

# Terrain generieren
result = agent.generate_terrain(
    output_folder=Path("./terrain/mountain"),
    scene_seed=42,
    terrain_type="mountain",
    detail_level="high"
)

# Terrain optimieren
result = agent.optimize_terrain(
    terrain_folder=Path("./terrain/mountain"),
    optimization_level="medium"
)
```

### **RenderControllerAgent**
**Dependencies:** Render + Core
**Tools:** Rendering, Ground Truth, Kamera-Setup
**Verwendung:**
```python
agent = system.render_controller

# Szene rendern
result = agent.render_scene(
    scene_folder=Path("./scenes/forest"),
    output_folder=Path("./renders/forest"),
    render_settings={
        "resolution": (1920, 1080),
        "samples": 256,
        "engine": "cycles"
    }
)

# Ground Truth generieren
result = agent.generate_ground_truth(
    scene_folder=Path("./scenes/forest"),
    output_folder=Path("./gt/forest"),
    gt_types=["depth", "normal", "segmentation"]
)
```

### **DataManagerAgent**
**Dependencies:** Data + Core
**Tools:** Job-Management, Monitoring
**Verwendung:**
```python
agent = system.data_manager

# Job erstellen
result = agent.create_data_generation_job(
    job_name="nature_dataset",
    output_folder=Path("./datasets/nature"),
    scene_seeds=[42, 123, 456],
    tasks=["coarse", "populate", "render"]
)

# Job überwachen
result = agent.monitor_job_progress("nature_dataset", detailed=True)
```

### **ExportSpecialistAgent**
**Dependencies:** Export + Core
**Tools:** Format-Konvertierung, Export-Optimierung
**Verwendung:**
```python
agent = system.export_specialist

# Szene exportieren
result = agent.export_scene_data(
    input_blend_file=Path("./scenes/forest/scene.blend"),
    output_folder=Path("./exports/forest"),
    export_formats=["obj", "fbx", "usdc"]
)

# Mesh konvertieren
result = agent.convert_mesh_format(
    input_file=Path("./meshes/tree.obj"),
    output_file=Path("./meshes/tree.fbx"),
    target_format="fbx"
)
```

## 📊 **Empfohlene Agent-Kombinationen**

### **Minimal Setup (3 Agents)**
- SceneComposerAgent
- AssetGeneratorAgent  
- RenderControllerAgent

### **Standard Setup (5 Agents)**
- SceneComposerAgent
- AssetGeneratorAgent
- TerrainEngineerAgent
- RenderControllerAgent
- DataManagerAgent

### **Vollständiges Setup (6 Agents)**
- Alle Agents für komplette Pipeline

## 🔧 **Konfiguration**

### **Dependency-Management**
Jeder Agent lädt nur die benötigten Dependencies:
```python
# Nur Core-Dependencies
core_deps = CoreDependencies()

# Blender + Core
blender_deps = BlenderDependencies()

# Terrain + Core
terrain_deps = TerrainDependencies()
```

### **Tool-Spezialisierung**
Tools sind auf spezifische Aufgaben fokussiert:
```python
# Nur Szenen-Tools
scene_tools = SceneTools()

# Nur Asset-Tools
asset_tools = AssetTools()
```

## 🎯 **Vorteile der Architektur**

1. **Minimale Dependencies**: Jeder Agent lädt nur was er braucht
2. **Fokussierte Tools**: Spezialisierte Tools für spezifische Aufgaben
3. **Modulare Architektur**: Agents können unabhängig verwendet werden
4. **AI-Integration**: Pydantic-AI für intelligente Entscheidungen
5. **Skalierbarkeit**: Einfach neue Agents hinzufügen
6. **Wartbarkeit**: Klare Trennung der Verantwortlichkeiten

## 🚀 **Nächste Schritte**

1. **Agent testen**: Einzelne Agents testen
2. **Pipeline aufbauen**: Agents kombinieren
3. **Erweitern**: Neue Agents hinzufügen
4. **Optimieren**: Performance verbessern
5. **Deployen**: In Produktion einsetzen

## 📝 **Beispiel-Workflow**

```python
# 1. System initialisieren
system = InfinigenAISystem()

# 2. Szene komponieren
scene_result = system.scene_composer.compose_nature_scene(
    output_folder=Path("./scenes/forest"),
    scene_seed=42,
    scene_type="forest"
)

# 3. Terrain generieren
terrain_result = system.terrain_engineer.generate_terrain(
    output_folder=Path("./scenes/forest"),
    scene_seed=42,
    terrain_type="mountain"
)

# 4. Assets generieren
asset_result = system.asset_generator.generate_tree_asset(
    tree_type="pine",
    output_path=Path("./assets/pine"),
    seed=42
)

# 5. Rendern
render_result = system.render_controller.render_scene(
    scene_folder=Path("./scenes/forest"),
    output_folder=Path("./renders/forest")
)

# 6. Exportieren
export_result = system.export_specialist.export_scene_data(
    input_blend_file=Path("./scenes/forest/scene.blend"),
    output_folder=Path("./exports/forest")
)
```

Dieses System folgt dem "Weniger ist mehr" Prinzip und bietet maximale Flexibilität bei minimaler Komplexität.
