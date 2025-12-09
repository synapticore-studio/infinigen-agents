#!/usr/bin/env python3
"""Test Orchestrator Agent"""

import asyncio
from deps.knowledge_deps import KnowledgeBase
from tools.ast_udfs import ASTUDFManager
from agents.orchestrator_agent import OrchestratorAgent

async def test():
    print('🎭 TESTING ORCHESTRATOR AGENT')
    print('=' * 50)
    
    kb = KnowledgeBase()
    ast_manager = ASTUDFManager(kb)
    orchestrator = OrchestratorAgent(kb, ast_manager)
    
    print('✅ Components initialized')
    
    result = await orchestrator.orchestrate_scene_generation(
        scene_type='forest',
        scene_seed=42,
        complexity='medium',
        quality='high'
    )
    
    success_status = 'Success' if result.get('success') else 'Failed'
    print(f'✅ Workflow Status: {success_status}')
    print(f'📊 Execution Time: {result.get("execution_time", 0):.2f}s')
    
    if result.get('workflow_state'):
        state = result['workflow_state']
        print(f'🎯 Scene Type: {state.scene_type}')
        print(f'🎲 Scene Seed: {state.scene_seed}')
        print(f'⚙️ Complexity: {state.complexity}')
        print(f'🎨 Quality: {state.quality}')
        print(f'📍 Current Step: {state.current_step}')
        print(f'✅ Success: {state.success}')
        if state.error_message:
            print(f'❌ Error: {state.error_message}')
    
    kb.close()
    print('\n✅ Test completed successfully!')

if __name__ == "__main__":
    asyncio.run(test())
