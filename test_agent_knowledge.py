#!/usr/bin/env python3
"""Test the Infinigen Agent Knowledge System"""

import logging
from pathlib import Path

from deps.knowledge_deps import KnowledgeBase
from infinigen_agent_knowledge import InfinigenAgentKnowledge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_knowledge():
    """Test the agent knowledge system"""

    # Initialize knowledge base
    kb = KnowledgeBase(db_path=Path("./infinigen_agent_knowledge.db"))
    agent_kb = InfinigenAgentKnowledge(kb)

    print("🎯 AGENT-SPEZIFISCHE WISSENSTESTS:")
    print("=" * 50)

    # Test different agent types with specific knowledge
    agents_queries = [
        ("scene_composer", "Task.Coarse Task.Populate scene composition"),
        ("asset_generator", "Factory classes RockyPlanetFactory GasGiantFactory"),
        ("terrain_engineer", "LandLab terrain generation noise patterns"),
        ("render_controller", "BLENDER_EEVEE_NEXT CYCLES render engines"),
        ("data_manager", "export formats pipeline tasks ground truth"),
        ("export_specialist", "MJCF URDF simulator export tools"),
        ("addon_manager", "Blender MaterialX OpenImageIO addon system"),
    ]

    for agent, query in agents_queries:
        print(f"\n🎬 {agent.upper().replace('_', ' ')}:")
        results = agent_kb.query_agent_knowledge(agent, query, limit=3)
        print(f"  Query: '{query}' -> {len(results)} results")

        for i, result in enumerate(results):
            scene_type = result.get("scene_type", "unknown")
            subcategory = result.get("parameters", {}).get("subcategory", "unknown")
            params = result.get("parameters", {})

            print(f"    {i+1}. {scene_type} - {subcategory}")

            # Show specific knowledge for each agent
            if agent == "scene_composer" and "task_types" in params:
                print(f"      ✅ Task Types: {params['task_types']}")
            elif agent == "asset_generator" and "factory_classes" in params:
                print(f"      ✅ Factory Classes: {params['factory_classes']}")
            elif agent == "terrain_engineer" and "terrain_system" in params:
                print(f"      ✅ Terrain System: {params['terrain_system']}")
            elif agent == "render_controller" and "render_engines" in params:
                print(f"      ✅ Render Engines: {params['render_engines']}")
            elif agent == "data_manager" and "export_formats" in params:
                print(f"      ✅ Export Formats: {params['export_formats']}")
            elif agent == "export_specialist" and "simulator_formats" in params:
                print(f"      ✅ Simulator Formats: {params['simulator_formats']}")
            elif agent == "addon_manager" and "blender_integration" in params:
                print(f"      ✅ Blender Integration: {params['blender_integration']}")

    # Test semantic search directly
    print(f"\n🔍 Direct semantic search:")
    direct_results = kb.semantic_search_scenes("astronomy planets factories", limit=3)
    print(f"  Found {len(direct_results)} results")
    for i, result in enumerate(direct_results):
        print(
            f"    {i+1}. {result['scene_type']} (similarity: {result['similarity']:.3f})"
        )

    kb.close()
    print("\n✅ Alle Agents haben ihr spezifisches Wissen!")


if __name__ == "__main__":
    test_agent_knowledge()
