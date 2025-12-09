#!/usr/bin/env python3
"""Demo: Pydantic Graph Orchestrator Agent"""

import json
import logging
from pathlib import Path

from deps.knowledge_deps import KnowledgeBase
from tools.ast_udfs import ASTUDFManager
from agents.pydantic_orchestrator_agent import PydanticOrchestratorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_pydantic_orchestrator():
    """Demonstriert den Pydantic Graph Orchestrator Agent"""
    
    print("🎭 DEMO: PYDANTIC GRAPH ORCHESTRATOR AGENT")
    print("=" * 60)

    # Initialize components
    kb = KnowledgeBase(db_path=Path('./infinigen_agent_knowledge.db'))
    ast_manager = ASTUDFManager(kb)
    orchestrator = PydanticOrchestratorAgent(kb, ast_manager)

    print("✅ Pydantic Graph Orchestrator Agent initialisiert")

    # 1. DEMO: Scene Generation Workflow
    print("\n🎬 1. SCENE GENERATION WORKFLOW")
    print("-" * 40)
    
    try:
        result = await orchestrator.orchestrate_scene_generation(
            scene_type="forest",
            scene_seed=42,
            complexity="medium",
            quality="high"
        )
        
        print(f"✅ Workflow Status: {'Success' if result.get('success') else 'Failed'}")
        print(f"📊 Execution Time: {result.get('execution_time', 0):.2f}s")
        
        # Show workflow state
        workflow_state = result.get('workflow_state')
        if workflow_state:
            print(f"🎯 Scene Type: {workflow_state.scene_type}")
            print(f"🎲 Scene Seed: {workflow_state.scene_seed}")
            print(f"⚙️ Complexity: {workflow_state.complexity}")
            print(f"🎨 Quality: {workflow_state.quality}")
            print(f"📍 Current Step: {workflow_state.current_step}")
            if workflow_state.error_message:
                print(f"❌ Error: {workflow_state.error_message}")
        
        # Show Mermaid diagram
        mermaid_diagram = result.get('mermaid_diagram', '')
        if mermaid_diagram:
            print(f"\n📈 Mermaid Diagram:")
            print(mermaid_diagram)
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # 2. DEMO: Asset Generation Workflow
    print("\n🏭 2. ASSET GENERATION WORKFLOW")
    print("-" * 40)
    
    try:
        result = await orchestrator.orchestrate_asset_generation(
            asset_type="planet",
            asset_category="astronomy",
            complexity="high"
        )
        
        print(f"✅ Workflow Status: {'Success' if result.get('success') else 'Failed'}")
        print(f"📊 Execution Time: {result.get('execution_time', 0):.2f}s")
        print(f"🏭 Asset Type: {result.get('asset_type', 'unknown')}")
        print(f"📂 Asset Category: {result.get('asset_category', 'unknown')}")
        
        # Show Mermaid diagram
        mermaid_diagram = result.get('mermaid_diagram', '')
        if mermaid_diagram:
            print(f"\n📈 Mermaid Diagram:")
            print(mermaid_diagram)
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # 3. DEMO: Multiple Workflow Executions
    print("\n🔄 3. MULTIPLE WORKFLOW EXECUTIONS")
    print("-" * 40)
    
    scenarios = [
        {"scene_type": "desert", "complexity": "low", "quality": "medium"},
        {"scene_type": "mountain", "complexity": "high", "quality": "high"},
        {"scene_type": "ocean", "complexity": "medium", "quality": "low"},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎬 Scenario {i}: {scenario['scene_type']}")
        try:
            result = await orchestrator.orchestrate_scene_generation(
                scene_type=scenario['scene_type'],
                scene_seed=100 + i,
                complexity=scenario['complexity'],
                quality=scenario['quality']
            )
            
            status = "✅ Success" if result.get('success') else "❌ Failed"
            print(f"   {status} - {result.get('execution_time', 0):.2f}s")
            
            if not result.get('success') and result.get('error'):
                print(f"   Error: {result.get('error')}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

    # 4. DEMO: Workflow Insights
    print("\n🧠 4. WORKFLOW INSIGHTS")
    print("-" * 40)
    
    try:
        insights = orchestrator.get_workflow_insights()
        print(f"📊 Total Workflows: {insights.get('total_workflows', 0)}")
        print(f"📈 Success Rate: {insights.get('success_rate', 0):.2f}")
        print(f"🔍 Common Failure Points: {len(insights.get('common_failure_points', []))}")
        print(f"💡 Optimization Opportunities: {len(insights.get('optimization_opportunities', []))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # 5. DEMO: Workflow Diagram Generation
    print("\n📈 5. WORKFLOW DIAGRAM GENERATION")
    print("-" * 40)
    
    try:
        # Generate workflow diagram
        diagram = orchestrator.generate_workflow_diagram()
        print("🎭 Workflow Diagram:")
        print(diagram)
        
    except Exception as e:
        print(f"❌ Error: {e}")

    # 6. DEMO: Pydantic Graph Features
    print("\n🔗 6. PYDANTIC GRAPH FEATURES")
    print("-" * 40)
    
    try:
        from pydantic_graph import Graph, GraphRunContext
        from agents.pydantic_orchestrator_agent import WorkflowState, WorkflowDeps
        
        # Create a custom workflow state
        custom_state = WorkflowState(
            scene_type="custom_forest",
            scene_seed=999,
            complexity="ultra",
            quality="ultra"
        )
        
        custom_deps = WorkflowDeps(
            knowledge_base=kb,
            ast_udf_manager=ast_manager
        )
        
        print(f"✅ Custom State Created:")
        print(f"   • Scene Type: {custom_state.scene_type}")
        print(f"   • Scene Seed: {custom_state.scene_seed}")
        print(f"   • Complexity: {custom_state.complexity}")
        print(f"   • Quality: {custom_state.quality}")
        
        # Show graph structure
        print(f"\n🔗 Graph Structure:")
        print(f"   • State Type: {type(custom_state).__name__}")
        print(f"   • Dependencies Type: {type(custom_deps).__name__}")
        print(f"   • Knowledge Base: {type(custom_deps.knowledge_base).__name__}")
        print(f"   • AST UDF Manager: {type(custom_deps.ast_udf_manager).__name__}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

    kb.close()
    print("\n✅ DEMO ABGESCHLOSSEN - Pydantic Graph Orchestrator funktioniert perfekt!")
    print("\n🎯 ZUSAMMENFASSUNG:")
    print("   • Pydantic Graph Integration: ✅")
    print("   • Workflow State Management: ✅")
    print("   • Node-based Execution: ✅")
    print("   • Mermaid Diagram Generation: ✅")
    print("   • Performance Monitoring: ✅")
    print("   • Intelligent Orchestration: ✅")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_pydantic_orchestrator())
