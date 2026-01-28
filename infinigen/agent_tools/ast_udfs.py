# AST UDFs for DuckDB - Intelligent Code Analysis
import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb

from infinigen.agent_deps.knowledge_deps import KnowledgeBaseDep

# Simple dependency injection


@dataclass
class ASTUDFManager:
    """AST-based UDFs for intelligent code analysis and optimization"""

    def __init__(self, knowledge_base: KnowledgeBaseDep):
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)

        # Register UDFs with DuckDB
        self._register_udfs()

    def _register_udfs(self):
        """Register custom UDFs with DuckDB"""

        # Register AST analysis UDF
        duckdb.create_function(
            "analyze_code_complexity",
            self._analyze_code_complexity,
            ["VARCHAR"],
            "VARCHAR",
        )

        # Register parameter optimization UDF
        duckdb.create_function(
            "optimize_parameters",
            self._optimize_parameters,
            ["VARCHAR", "VARCHAR"],
            "VARCHAR",
        )

        # Register error pattern detection UDF
        duckdb.create_function(
            "detect_error_patterns",
            self._detect_error_patterns,
            ["VARCHAR", "VARCHAR"],
            "VARCHAR",
        )

        # Register performance prediction UDF
        duckdb.create_function(
            "predict_performance",
            self._predict_performance,
            ["VARCHAR", "VARCHAR"],
            "VARCHAR",
        )

    def _analyze_code_complexity(self, code: str) -> str:
        """Analyze code complexity using AST"""
        try:
            tree = ast.parse(code)

            complexity_metrics = {
                "cyclomatic_complexity": 0,
                "function_count": 0,
                "class_count": 0,
                "import_count": 0,
                "nested_depth": 0,
                "line_count": len(code.splitlines()),
                "character_count": len(code),
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity_metrics["function_count"] += 1
                    complexity_metrics[
                        "cyclomatic_complexity"
                    ] += self._count_decision_points(node)

                elif isinstance(node, ast.ClassDef):
                    complexity_metrics["class_count"] += 1

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    complexity_metrics["import_count"] += 1

                elif isinstance(node, ast.For):
                    complexity_metrics["nested_depth"] = max(
                        complexity_metrics["nested_depth"],
                        self._get_nesting_depth(node),
                    )

            return json.dumps(complexity_metrics)

        except SyntaxError as e:
            return json.dumps({"error": f"Syntax error: {e}", "complexity": 0})
        except Exception as e:
            return json.dumps({"error": f"Analysis error: {e}", "complexity": 0})

    def _count_decision_points(self, node: ast.AST) -> int:
        """Count decision points in a function for cyclomatic complexity"""
        count = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                count += 1
            elif isinstance(child, ast.BoolOp):
                count += len(child.values) - 1

        return count

    def _get_nesting_depth(
        self, node: ast.AST, current_depth: int = 0, max_depth_limit: int = 10
    ) -> int:
        """Get maximum nesting depth in code with recursion limit"""
        if current_depth > max_depth_limit:
            return current_depth

        max_depth = current_depth

        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.Try)):
                child_depth = self._get_nesting_depth(
                    child, current_depth + 1, max_depth_limit
                )
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _optimize_parameters(self, agent_name: str, current_params_json: str) -> str:
        """Optimize parameters based on historical knowledge"""

        try:
            current_params = json.loads(current_params_json)
        except (json.JSONDecodeError, TypeError):
            return current_params_json

        # Get similar successful cases
        similar_cases = self.knowledge_base.get_similar_successful_cases(
            agent_name, current_params, limit=10
        )

        if not similar_cases:
            return json.dumps(current_params)

        # Analyze parameter patterns
        optimized_params = current_params.copy()

        # Find most successful parameter values
        param_success_rates = {}
        for case in similar_cases:
            for param, value in case["parameters"].items():
                if param not in param_success_rates:
                    param_success_rates[param] = {}

                value_key = str(value)
                if value_key not in param_success_rates[param]:
                    param_success_rates[param][value_key] = {"count": 0, "success": 0}

                param_success_rates[param][value_key]["count"] += 1
                param_success_rates[param][value_key]["success"] += 1

        # Optimize parameters based on success rates
        for param, value in current_params.items():
            if param in param_success_rates:
                best_value = max(
                    param_success_rates[param].items(),
                    key=lambda x: x[1]["success"] / x[1]["count"],
                )[0]

                # Convert back to original type
                try:
                    if isinstance(value, int):
                        optimized_params[param] = int(best_value)
                    elif isinstance(value, float):
                        optimized_params[param] = float(best_value)
                    elif isinstance(value, bool):
                        optimized_params[param] = best_value.lower() == "true"
                    else:
                        optimized_params[param] = best_value
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass

        return json.dumps(optimized_params)

    def _detect_error_patterns(self, agent_name: str, current_params_json: str) -> str:
        """Detect potential error patterns based on historical data"""

        try:
            current_params = json.loads(current_params_json)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "Invalid JSON parameters"})

        # Get agent performance data
        performance = self.knowledge_base.get_best_practices(agent_name, "general")

        error_patterns = {
            "common_errors": performance.get("common_errors", []),
            "risk_factors": [],
            "recommendations": [],
        }

        # Analyze current parameters for risk factors
        for param, value in current_params.items():
            # Check for common problematic values
            if param == "complexity" and value == "high":
                error_patterns["risk_factors"].append(
                    "High complexity may cause performance issues"
                )
                error_patterns["recommendations"].append(
                    "Consider using 'medium' complexity for better stability"
                )

            elif param == "seed" and value == 0:
                error_patterns["risk_factors"].append(
                    "Seed 0 may cause deterministic issues"
                )
                error_patterns["recommendations"].append(
                    "Use a non-zero seed for better randomization"
                )

            elif param == "quality" and value == "low":
                error_patterns["risk_factors"].append(
                    "Low quality may result in poor output"
                )
                error_patterns["recommendations"].append(
                    "Consider using 'medium' or 'high' quality"
                )

        return json.dumps(error_patterns)

    def _predict_performance(self, agent_name: str, current_params_json: str) -> str:
        """Predict performance based on historical data"""

        try:
            current_params = json.loads(current_params_json)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "Invalid JSON parameters"})

        # Get similar cases for performance prediction
        similar_cases = self.knowledge_base.get_similar_successful_cases(
            agent_name, current_params, limit=20
        )

        if not similar_cases:
            return json.dumps(
                {
                    "predicted_execution_time": 0.0,
                    "confidence": 0.0,
                    "success_probability": 0.5,
                }
            )

        # Calculate average performance metrics
        total_time = 0.0
        success_count = 0

        for case in similar_cases:
            metrics = case.get("performance_metrics", {})
            if "execution_time" in metrics:
                total_time += metrics["execution_time"]
            success_count += 1

        avg_execution_time = total_time / len(similar_cases) if similar_cases else 0.0
        success_rate = success_count / len(similar_cases) if similar_cases else 0.5

        # Calculate confidence based on similarity
        avg_similarity = sum(case["similarity"] for case in similar_cases) / len(
            similar_cases
        )
        confidence = min(avg_similarity * 2, 1.0)  # Scale similarity to confidence

        return json.dumps(
            {
                "predicted_execution_time": avg_execution_time,
                "confidence": confidence,
                "success_probability": success_rate,
                "sample_size": len(similar_cases),
            }
        )

    def analyze_agent_code(self, agent_code: str) -> Dict[str, Any]:
        """Analyze agent code for optimization opportunities"""

        complexity_json = self._analyze_code_complexity(agent_code)
        complexity = json.loads(complexity_json)

        analysis = {
            "complexity_metrics": complexity,
            "optimization_suggestions": [],
            "performance_concerns": [],
        }

        # Generate optimization suggestions
        if complexity.get("cyclomatic_complexity", 0) > 10:
            analysis["optimization_suggestions"].append(
                "High cyclomatic complexity detected. Consider breaking down complex functions."
            )

        if complexity.get("nested_depth", 0) > 3:
            analysis["optimization_suggestions"].append(
                "Deep nesting detected. Consider using early returns or guard clauses."
            )

        if complexity.get("function_count", 0) > 20:
            analysis["optimization_suggestions"].append(
                "Large number of functions. Consider splitting into multiple classes."
            )

        return analysis


# Simple dependency injection
ASTUDFManagerDep = ASTUDFManager
