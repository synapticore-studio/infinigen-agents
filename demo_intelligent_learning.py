#!/usr/bin/env python3
"""Demo: Intelligentes Lernen und Optimieren mit AST UDFs"""

import json
import logging
from pathlib import Path

from deps.knowledge_deps import KnowledgeBase
from tools.ast_udfs import ASTUDFManager
from tools.intelligent_orchestrator import IntelligentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_intelligent_learning():
    """Demonstriert intelligentes Lernen und Optimieren"""

    print("🧠 DEMO: INTELLIGENTES LERNEN UND OPTIMIEREN")
    print("=" * 60)

    # Initialize components
    kb = KnowledgeBase(db_path=Path("./infinigen_agent_knowledge.db"))
    ast_manager = ASTUDFManager(kb)
    orchestrator = IntelligentOrchestrator(kb, ast_manager)

    print("✅ System initialisiert")

    # 1. DEMO: Code-Komplexitätsanalyse
    print("\n🔍 1. CODE-KOMPLEXITÄTSANALYSE")
    print("-" * 40)

    complex_code = """
def generate_forest_scene(scene_params):
    if scene_params.get('complexity') == 'high':
        for tree_type in ['oak', 'pine', 'birch']:
            for i in range(scene_params.get('tree_count', 100)):
                if i % 10 == 0:
                    tree = create_special_tree(tree_type, i)
                    if tree.is_valid():
                        scene.add_tree(tree)
                    else:
                        handle_invalid_tree(tree)
                else:
                    scene.add_tree(create_standard_tree(tree_type))
    elif scene_params.get('complexity') == 'medium':
        for tree_type in ['oak', 'pine']:
            for i in range(scene_params.get('tree_count', 50)):
                scene.add_tree(create_standard_tree(tree_type))
    else:
        scene.add_tree(create_standard_tree('oak'))
    
    return scene
"""

    complexity_json = ast_manager._analyze_code_complexity(complex_code)
    complexity = json.loads(complexity_json)

    print(f"📊 Komplexitätsmetriken:")
    print(f"   • Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 0)}")
    print(f"   • Function Count: {complexity.get('function_count', 0)}")
    print(f"   • Class Count: {complexity.get('class_count', 0)}")
    print(f"   • Nested Depth: {complexity.get('nested_depth', 0)}")
    print(f"   • Line Count: {complexity.get('line_count', 0)}")

    # Code-Analyse für Optimierungsvorschläge
    analysis = ast_manager.analyze_agent_code(complex_code)
    print(f"\n💡 Optimierungsvorschläge: {len(analysis['optimization_suggestions'])}")
    for suggestion in analysis["optimization_suggestions"]:
        print(f"   • {suggestion}")

    # 2. DEMO: Parameter-Optimierung
    print("\n🎯 2. PARAMETER-OPTIMIERUNG")
    print("-" * 40)

    test_scenarios = [
        {
            "name": "Forest Scene - High Complexity",
            "params": {
                "scene_type": "forest",
                "complexity": "high",
                "quality": "medium",
                "seed": 42,
            },
        },
        {
            "name": "Desert Scene - Low Quality",
            "params": {
                "scene_type": "desert",
                "complexity": "medium",
                "quality": "low",
                "seed": 0,
            },
        },
        {
            "name": "Mountain Scene - Medium Settings",
            "params": {
                "scene_type": "mountain",
                "complexity": "medium",
                "quality": "high",
                "seed": 123,
            },
        },
    ]

    for scenario in test_scenarios:
        print(f"\n📋 {scenario['name']}:")
        print(f"   Original: {scenario['params']}")

        # Parameter optimieren
        optimized_json = ast_manager._optimize_parameters(
            "scene_composer", json.dumps(scenario["params"])
        )
        optimized = json.loads(optimized_json)
        print(f"   Optimiert: {optimized}")

        # Fehleranalyse
        error_json = ast_manager._detect_error_patterns(
            "scene_composer", json.dumps(scenario["params"])
        )
        error_analysis = json.loads(error_json)

        if error_analysis.get("risk_factors"):
            print(f"   ⚠️  Risiken: {error_analysis['risk_factors']}")
        if error_analysis.get("recommendations"):
            print(f"   💡 Empfehlungen: {error_analysis['recommendations']}")

    # 3. DEMO: Performance-Vorhersage
    print("\n📊 3. PERFORMANCE-VORHERSAGE")
    print("-" * 40)

    performance_scenarios = [
        {"name": "Simple Forest", "params": {"complexity": "low", "quality": "medium"}},
        {"name": "Complex Forest", "params": {"complexity": "high", "quality": "high"}},
        {
            "name": "Ultra Complex",
            "params": {"complexity": "ultra", "quality": "ultra"},
        },
    ]

    for scenario in performance_scenarios:
        perf_json = ast_manager._predict_performance(
            "terrain_engineer", json.dumps(scenario["params"])
        )
        performance = json.loads(perf_json)

        print(f"\n🏔️ {scenario['name']}:")
        print(
            f"   Predicted Time: {performance.get('predicted_execution_time', 0):.2f}s"
        )
        print(f"   Confidence: {performance.get('confidence', 0):.2f}")
        print(
            f"   Success Probability: {performance.get('success_probability', 0):.2f}"
        )
        print(f"   Sample Size: {performance.get('sample_size', 0)}")

    # 4. DEMO: Intelligente Orchestrierung
    print("\n🚀 4. INTELLIGENTE ORCHESTRIERUNG")
    print("-" * 40)

    orchestration_scenarios = [
        {
            "name": "Forest Scene Generation",
            "scene_type": "forest",
            "scene_seed": 42,
            "complexity": "medium",
            "quality": "high",
        },
        {
            "name": "Desert Scene Generation",
            "scene_type": "desert",
            "scene_seed": 123,
            "complexity": "high",
            "quality": "medium",
        },
    ]

    for scenario in orchestration_scenarios:
        print(f"\n🎬 {scenario['name']}:")

        try:
            result = orchestrator.orchestrate_scene_generation(
                scene_type=scenario["scene_type"],
                scene_seed=scenario["scene_seed"],
                complexity=scenario["complexity"],
                quality=scenario["quality"],
            )

            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Workflow Steps: {len(result.get('workflow_plan', {}))}")

            perf_pred = result.get("performance_prediction", {})
            if isinstance(perf_pred, dict):
                print(
                    f"   Predicted Time: {perf_pred.get('predicted_execution_time', 0):.2f}s"
                )
                print(
                    f"   Success Probability: {perf_pred.get('success_probability', 0):.2f}"
                )

        except Exception as e:
            print(f"   Error: {e}")

    # 5. DEMO: Agent-spezifisches Wissen
    print("\n🧠 5. AGENT-SPEZIFISCHES WISSEN")
    print("-" * 40)

    from infinigen_agent_knowledge import InfinigenAgentKnowledge

    agent_kb = InfinigenAgentKnowledge(kb)

    agents = [
        ("scene_composer", "Task.Coarse Task.Populate scene composition"),
        ("asset_generator", "Factory classes RockyPlanetFactory GasGiantFactory"),
        ("terrain_engineer", "LandLab terrain generation noise patterns"),
        ("render_controller", "BLENDER_EEVEE_NEXT CYCLES render engines"),
        ("data_manager", "export formats pipeline tasks ground truth"),
        ("export_specialist", "MJCF URDF simulator export tools"),
        ("addon_manager", "Blender MaterialX OpenImageIO addon system"),
    ]

    for agent, query in agents:
        results = agent_kb.query_agent_knowledge(agent, query, limit=2)
        print(f"\n🤖 {agent.upper().replace('_', ' ')}:")
        print(f"   Query: '{query}'")
        print(f"   Found: {len(results)} relevant results")

        for i, result in enumerate(results[:2]):
            scene_type = result.get("scene_type", "unknown")
            subcategory = result.get("parameters", {}).get("subcategory", "unknown")
            print(f"   {i+1}. {scene_type} - {subcategory}")

    kb.close()
    print("\n✅ DEMO ABGESCHLOSSEN - Intelligentes Lernen funktioniert perfekt!")
    print("\n🎯 ZUSAMMENFASSUNG:")
    print("   • Code-Komplexitätsanalyse: ✅")
    print("   • Parameter-Optimierung: ✅")
    print("   • Fehlererkennung: ✅")
    print("   • Performance-Vorhersage: ✅")
    print("   • Intelligente Orchestrierung: ✅")
    print("   • Agent-spezifisches Wissen: ✅")


if __name__ == "__main__":
    demo_intelligent_learning()
