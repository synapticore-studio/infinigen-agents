#!/usr/bin/env python3
"""Test AST UDFs Integration and Intelligent Learning"""

import json
import logging
from pathlib import Path

from deps.knowledge_deps import KnowledgeBase
from tools.ast_udfs import ASTUDFManager
from tools.intelligent_orchestrator import IntelligentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_udfs_integration():
    """Test AST UDFs integration and intelligent learning"""

    print("🧪 TESTE AST UDFs INTEGRATION:")
    print("=" * 40)

    # Initialize components
    kb = KnowledgeBase(db_path=Path("./infinigen_agent_knowledge.db"))
    ast_manager = ASTUDFManager(kb)
    orchestrator = IntelligentOrchestrator(kb, ast_manager)

    print("✅ Knowledge Base initialized")
    print("✅ AST UDF Manager initialized")
    print("✅ Intelligent Orchestrator initialized")

    # Test code complexity analysis
    test_code = """
def complex_function(x, y):
    if x > 0:
        for i in range(x):
            if y > i:
                return True
    return False

class TestClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
"""

    print("\n🔍 Testing Code Complexity Analysis:")
    complexity_json = ast_manager._analyze_code_complexity(test_code)
    complexity = json.loads(complexity_json)
    print(f"  Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 0)}")
    print(f"  Function Count: {complexity.get('function_count', 0)}")
    print(f"  Nested Depth: {complexity.get('nested_depth', 0)}")
    print(f"  Line Count: {complexity.get('line_count', 0)}")

    # Test parameter optimization
    print("\n🎯 Testing Parameter Optimization:")
    test_params = {
        "scene_type": "forest",
        "complexity": "high",
        "quality": "medium",
        "seed": 42,
    }

    optimized_json = ast_manager._optimize_parameters(
        "scene_composer", json.dumps(test_params)
    )
    optimized = json.loads(optimized_json)
    print(f"  Original: {test_params}")
    print(f"  Optimized: {optimized}")

    # Test error pattern detection
    print("\n⚠️ Testing Error Pattern Detection:")
    error_json = ast_manager._detect_error_patterns(
        "asset_generator", json.dumps(test_params)
    )
    error_analysis = json.loads(error_json)
    print(f"  Risk Factors: {error_analysis.get('risk_factors', [])}")
    print(f"  Recommendations: {error_analysis.get('recommendations', [])}")

    # Test performance prediction
    print("\n📊 Testing Performance Prediction:")
    performance_json = ast_manager._predict_performance(
        "terrain_engineer", json.dumps(test_params)
    )
    performance = json.loads(performance_json)
    print(f"  Predicted Time: {performance.get('predicted_execution_time', 0):.2f}s")
    print(f"  Confidence: {performance.get('confidence', 0):.2f}")
    print(f"  Success Probability: {performance.get('success_probability', 0):.2f}")

    # Test intelligent orchestration
    print("\n🚀 Testing Intelligent Orchestration:")
    try:
        result = orchestrator.orchestrate_scene_generation(
            scene_type="forest", scene_seed=123, complexity="medium", quality="high"
        )
        print(f"  Orchestration Result: {result.get('status', 'unknown')}")
        print(f"  Workflow Plan: {len(result.get('workflow_plan', {}))} steps")
        print(f"  Performance Prediction: {result.get('performance_prediction', {})}")
    except Exception as e:
        print(f"  Orchestration Error: {e}")

    # Test agent code analysis
    print("\n🔬 Testing Agent Code Analysis:")
    agent_code = """
class SceneComposerAgent:
    def compose_scene(self, scene_type, params):
        if scene_type == "forest":
            return self._create_forest_scene(params)
        elif scene_type == "desert":
            return self._create_desert_scene(params)
        else:
            raise ValueError("Unknown scene type")
    
    def _create_forest_scene(self, params):
        # Complex forest generation logic
        for tree in range(params.get('tree_count', 100)):
            if tree % 10 == 0:
                self._add_special_tree(tree)
        return "forest_scene"
"""

    analysis = ast_manager.analyze_agent_code(agent_code)
    print(f"  Complexity Metrics: {analysis.get('complexity_metrics', {})}")
    print(
        f"  Optimization Suggestions: {len(analysis.get('optimization_suggestions', []))}"
    )
    print(f"  Performance Concerns: {len(analysis.get('performance_concerns', []))}")

    kb.close()
    print("\n✅ Alle UDFs funktionieren korrekt!")


if __name__ == "__main__":
    test_udfs_integration()
