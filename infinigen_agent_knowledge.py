#!/usr/bin/env python3
"""
Infinigen Agent Knowledge System
Extracts and manages knowledge from real Infinigen configurations, docs, and workflows
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import gin

from deps.knowledge_deps import KnowledgeBase

logger = logging.getLogger(__name__)


class InfinigenAgentKnowledge:
    """Knowledge system specifically for Infinigen AI Agents"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.configs_path = Path("infinigen_examples")
        self.docs_path = Path("docs")

    def extract_gin_config_knowledge(self) -> List[Dict[str, Any]]:
        """Extract knowledge from Gin configuration files"""
        knowledge_entries = []

        # Scene Types Knowledge
        scene_types_path = self.configs_path / "configs_nature" / "scene_types"
        if scene_types_path.exists():
            for gin_file in scene_types_path.glob("*.gin"):
                scene_type = gin_file.stem
                knowledge = self._parse_gin_file(gin_file, "scene_type", scene_type)
                knowledge_entries.extend(knowledge)

        # Astronomy Configs
        astronomy_path = self.configs_path / "configs_astronomy"
        if astronomy_path.exists():
            for gin_file in astronomy_path.glob("*.gin"):
                asset_type = gin_file.stem
                knowledge = self._parse_gin_file(
                    gin_file, "astronomy_asset", asset_type
                )
                knowledge_entries.extend(knowledge)

        # Performance Configs
        performance_path = self.configs_path / "configs_nature" / "performance"
        if performance_path.exists():
            for gin_file in performance_path.glob("*.gin"):
                perf_type = gin_file.stem
                knowledge = self._parse_gin_file(gin_file, "performance", perf_type)
                knowledge_entries.extend(knowledge)

        # Indoor Configs
        indoor_path = self.configs_path / "configs_indoor"
        if indoor_path.exists():
            for gin_file in indoor_path.glob("*.gin"):
                indoor_type = gin_file.stem
                knowledge = self._parse_gin_file(gin_file, "indoor_config", indoor_type)
                knowledge_entries.extend(knowledge)

        return knowledge_entries

    def _parse_gin_file(
        self, gin_file: Path, category: str, subcategory: str
    ) -> List[Dict[str, Any]]:
        """Parse a Gin file and extract parameter knowledge"""
        knowledge_entries = []

        try:
            with open(gin_file, "r") as f:
                content = f.read()

            # Extract parameter patterns
            param_pattern = r"^([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*) = (.+)$"
            lines = content.split("\n")

            parameters = {}
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    match = re.match(param_pattern, line)
                    if match:
                        param_name = match.group(1)
                        param_value = match.group(2).strip()
                        parameters[param_name] = param_value

            if parameters:
                knowledge_entries.append(
                    {
                        "category": category,
                        "subcategory": subcategory,
                        "file_path": str(gin_file),
                        "parameters": parameters,
                        "parameter_count": len(parameters),
                        "complexity": self._assess_config_complexity(parameters),
                    }
                )

        except Exception as e:
            logger.warning(f"Could not parse {gin_file}: {e}")

        return knowledge_entries

    def _assess_config_complexity(self, parameters: Dict[str, str]) -> str:
        """Assess complexity of a configuration based on parameters"""
        param_count = len(parameters)

        # Count complex parameters (arrays, ranges, etc.)
        complex_params = 0
        for value in parameters.values():
            if any(x in value for x in ["[", "(", "uniform", "normal", "choice"]):
                complex_params += 1

        if param_count > 20 or complex_params > 5:
            return "high"
        elif param_count > 10 or complex_params > 2:
            return "medium"
        else:
            return "low"

    def extract_workflow_knowledge(self) -> List[Dict[str, Any]]:
        """Extract knowledge from workflow examples"""
        knowledge_entries = []

        # Astronomy Workflow
        astronomy_workflow = Path("infinigen_examples/astronomy_workflow.py")
        if astronomy_workflow.exists():
            workflow_knowledge = self._parse_workflow_file(
                astronomy_workflow, "astronomy"
            )
            knowledge_entries.extend(workflow_knowledge)

        # Nature Generation
        nature_workflow = Path("infinigen_examples/generate_nature.py")
        if nature_workflow.exists():
            workflow_knowledge = self._parse_workflow_file(nature_workflow, "nature")
            knowledge_entries.extend(workflow_knowledge)

        return knowledge_entries

    def _parse_workflow_file(
        self, workflow_file: Path, workflow_type: str
    ) -> List[Dict[str, Any]]:
        """Parse workflow file and extract patterns"""
        knowledge_entries = []

        try:
            with open(workflow_file, "r") as f:
                content = f.read()

            # Extract class definitions
            class_pattern = r"class\s+(\w+).*?:"
            classes = re.findall(class_pattern, content)

            # Extract method definitions
            method_pattern = r"def\s+(\w+).*?:"
            methods = re.findall(method_pattern, content)

            # Extract factory imports
            factory_pattern = r"from.*?\.(\w+Factory)"
            factories = re.findall(factory_pattern, content)

            # Extract task patterns
            task_pattern = r"Task\.(\w+)"
            tasks = re.findall(task_pattern, content)

            knowledge_entries.append(
                {
                    "category": "workflow",
                    "subcategory": workflow_type,
                    "file_path": str(workflow_file),
                    "classes": classes,
                    "methods": methods,
                    "factories": factories,
                    "tasks": tasks,
                    "complexity": "high" if len(classes) > 3 else "medium",
                }
            )

        except Exception as e:
            logger.warning(f"Could not parse workflow {workflow_file}: {e}")

        return knowledge_entries

    def extract_documentation_knowledge(self) -> List[Dict[str, Any]]:
        """Extract knowledge from documentation files"""
        knowledge_entries = []

        doc_files = [
            "ConfiguringInfinigen.md",
            "HelloWorld.md",
            "ImplementingAssets.md",
            "ConfiguringCameras.md",
            "ExportingToExternalFileFormats.md",
            "ExportingToSimulators.md",
            "GeneratingFluidSimulations.md",
            "GeneratingIndividualAssets.md",
            "GroundTruthAnnotations.md",
        ]

        for doc_file in doc_files:
            doc_path = self.docs_path / doc_file
            if doc_path.exists():
                doc_knowledge = self._parse_documentation(doc_path)
                knowledge_entries.extend(doc_knowledge)

        return knowledge_entries

    def _parse_documentation(self, doc_file: Path) -> List[Dict[str, Any]]:
        """Parse documentation and extract key concepts"""
        knowledge_entries = []

        try:
            with open(doc_file, "r") as f:
                content = f.read()

            # Extract code examples
            code_blocks = re.findall(
                r"```(?:bash|python)?\n(.*?)\n```", content, re.DOTALL
            )

            # Extract parameter examples
            param_examples = re.findall(
                r"`([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)`", content
            )

            # Extract task mentions
            task_mentions = re.findall(
                r"`(coarse|populate|fine_terrain|render)`", content
            )

            # Extract configuration mentions
            config_mentions = re.findall(r"`([a-z_]+\.gin)`", content)

            knowledge_entries.append(
                {
                    "category": "documentation",
                    "subcategory": doc_file.stem,
                    "file_path": str(doc_file),
                    "code_examples": len(code_blocks),
                    "parameter_examples": list(set(param_examples)),
                    "task_mentions": list(set(task_mentions)),
                    "config_mentions": list(set(config_mentions)),
                    "complexity": "high" if len(code_blocks) > 5 else "medium",
                }
            )

        except Exception as e:
            logger.warning(f"Could not parse documentation {doc_file}: {e}")

        return knowledge_entries

    def extract_infinigen_api_knowledge(self) -> List[Dict[str, Any]]:
        """Extract specific Infinigen API knowledge for each agent"""
        knowledge_entries = []

        # Scene Composer Knowledge
        scene_composer_knowledge = {
            "category": "api_knowledge",
            "subcategory": "scene_composer",
            "file_path": "infinigen/core/util/organization.py",
            "parameters": {
                "task_types": ["Task.Coarse", "Task.Populate", "Task.FineTerrain", "Task.Render"],
                "scene_types": ["forest", "desert", "mountain", "cave", "coral_reef", "arctic"],
                "composition_functions": ["compose_nature", "compose_indoors", "compose_astronomy"],
                "placement_system": "placement.populate_all",
                "semantic_tags": ["Semantics.Kitchen", "Semantics.Planet", "Semantics.StarField"]
            },
            "performance_metrics": {
                "complexity_score": 0.8,
                "api_coverage": 0.9
            },
            "generated_assets": ["Task", "Semantics", "compose_nature"],
            "complexity": "high"
        }
        knowledge_entries.append(scene_composer_knowledge)

        # Asset Generator Knowledge
        asset_generator_knowledge = {
            "category": "api_knowledge", 
            "subcategory": "asset_generator",
            "file_path": "infinigen/assets/",
            "parameters": {
                "factory_classes": ["RockyPlanetFactory", "GasGiantFactory", "TreeFactory", "RockFactory"],
                "asset_categories": ["astronomy", "vegetation", "terrain", "creatures"],
                "generation_methods": ["create_asset", "apply", "add_material"],
                "node_system": "Nodes.MeshCube, Nodes.MeshSphere, Nodes.MeshIcosphere",
                "material_system": "MaterialX, procedural materials"
            },
            "performance_metrics": {
                "complexity_score": 0.9,
                "api_coverage": 0.85
            },
            "generated_assets": ["Factory", "Nodes", "MaterialX"],
            "complexity": "high"
        }
        knowledge_entries.append(asset_generator_knowledge)

        # Terrain Engineer Knowledge
        terrain_engineer_knowledge = {
            "category": "api_knowledge",
            "subcategory": "terrain_engineer", 
            "file_path": "infinigen/terrain/",
            "parameters": {
                "terrain_system": "LandLab integration",
                "terrain_types": ["mountain", "plain", "canyon", "coast", "river"],
                "generation_methods": ["generate_terrain", "add_erosion", "add_vegetation"],
                "detail_levels": ["coarse", "medium", "fine", "ultra_fine"],
                "noise_patterns": ["perlin", "simplex", "voronoi", "fractal"]
            },
            "performance_metrics": {
                "complexity_score": 0.85,
                "api_coverage": 0.8
            },
            "generated_assets": ["LandLab", "terrain", "noise"],
            "complexity": "high"
        }
        knowledge_entries.append(terrain_engineer_knowledge)

        # Render Controller Knowledge
        render_controller_knowledge = {
            "category": "api_knowledge",
            "subcategory": "render_controller",
            "file_path": "infinigen/core/util/blender.py",
            "parameters": {
                "render_engines": ["BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"],
                "quality_settings": ["low", "medium", "high", "ultra"],
                "sampling_methods": ["adaptive", "fixed", "progressive"],
                "lighting_systems": ["sky_lighting", "sun_lighting", "artificial_lighting"],
                "camera_types": ["perspective", "orthographic", "panoramic"]
            },
            "performance_metrics": {
                "complexity_score": 0.7,
                "api_coverage": 0.9
            },
            "generated_assets": ["render", "lighting", "camera"],
            "complexity": "medium"
        }
        knowledge_entries.append(render_controller_knowledge)

        # Data Manager Knowledge
        data_manager_knowledge = {
            "category": "api_knowledge",
            "subcategory": "data_manager",
            "file_path": "infinigen/core/util/exporting.py",
            "parameters": {
                "export_formats": ["OBJ", "USD", "USDC", "FBX", "GLTF"],
                "data_types": ["meshes", "materials", "textures", "animations"],
                "pipeline_tasks": ["coarse", "populate", "fine_terrain", "render"],
                "file_management": ["scene.blend", "saved_mesh.json", "info.pickle"],
                "ground_truth": ["depth", "normal", "segmentation", "optical_flow"]
            },
            "performance_metrics": {
                "complexity_score": 0.6,
                "api_coverage": 0.8
            },
            "generated_assets": ["export", "pipeline", "ground_truth"],
            "complexity": "medium"
        }
        knowledge_entries.append(data_manager_knowledge)

        # Export Specialist Knowledge
        export_specialist_knowledge = {
            "category": "api_knowledge",
            "subcategory": "export_specialist",
            "file_path": "infinigen/tools/export/",
            "parameters": {
                "export_tools": ["export_scene", "triangulate_meshes", "export_to_simulator"],
                "simulator_formats": ["MJCF", "URDF", "SDF"],
                "optimization_methods": ["mesh_decimation", "texture_compression", "LOD_generation"],
                "validation_checks": ["mesh_integrity", "material_consistency", "scale_validation"]
            },
            "performance_metrics": {
                "complexity_score": 0.7,
                "api_coverage": 0.85
            },
            "generated_assets": ["export_tools", "simulator", "validation"],
            "complexity": "medium"
        }
        knowledge_entries.append(export_specialist_knowledge)

        # Addon Manager Knowledge
        addon_manager_knowledge = {
            "category": "api_knowledge",
            "subcategory": "addon_manager",
            "file_path": "infinigen/launch_blender.py",
            "parameters": {
                "blender_integration": ["launch_blender", "expose_bundled_modules"],
                "addon_system": ["MaterialX", "OpenImageIO", "VFX_libraries"],
                "version_management": ["Blender 4.4+", "Python 3.11", "compatibility_matrix"],
                "dependency_management": ["uv", "pip", "conda"]
            },
            "performance_metrics": {
                "complexity_score": 0.5,
                "api_coverage": 0.9
            },
            "generated_assets": ["addon", "blender", "dependencies"],
            "complexity": "low"
        }
        knowledge_entries.append(addon_manager_knowledge)

        return knowledge_entries

    def store_infinigen_knowledge(self):
        """Store all extracted Infinigen knowledge in the knowledge base"""

        logger.info("🔍 Extracting Infinigen knowledge...")

        # Extract from different sources
        gin_knowledge = self.extract_gin_config_knowledge()
        workflow_knowledge = self.extract_workflow_knowledge()
        doc_knowledge = self.extract_documentation_knowledge()
        api_knowledge = self.extract_infinigen_api_knowledge()

        all_knowledge = gin_knowledge + workflow_knowledge + doc_knowledge + api_knowledge

        logger.info(f"📊 Extracted {len(all_knowledge)} knowledge entries")

        # Store in knowledge base
        for entry in all_knowledge:
            try:
                # Create a context for semantic search
                context = f"{entry['category']} {entry['subcategory']} {entry.get('file_path', '')}"

                # Store as scene knowledge (reusing existing structure)
                scene_id = self.kb.store_scene_knowledge(
                    scene_type=f"infinigen_{entry['category']}",
                    scene_seed=hash(entry["file_path"]) % 10000,
                    success=True,
                    parameters={
                        "category": entry["category"],
                        "subcategory": entry["subcategory"],
                        "file_path": entry["file_path"],
                        "complexity": entry.get("complexity", "medium"),
                        **{
                            k: v
                            for k, v in entry.items()
                            if k
                            not in [
                                "category",
                                "subcategory",
                                "file_path",
                                "complexity",
                            ]
                        },
                    },
                    performance_metrics={
                        "parameter_count": entry.get("parameter_count", 0),
                        "code_examples": entry.get("code_examples", 0),
                        "complexity_score": (
                            1.0 if entry.get("complexity") == "high" else 0.5
                        ),
                    },
                    generated_assets=entry.get("factories", [])
                    + entry.get("classes", []),
                    error_messages=None,
                )

                logger.info(
                    f"✅ Stored {entry['category']}/{entry['subcategory']} knowledge (ID: {scene_id})"
                )

            except Exception as e:
                logger.error(f"Failed to store knowledge entry: {e}")

    def query_agent_knowledge(
        self, agent_type: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query knowledge specific to an agent type"""

        # Add agent context to query
        agent_query = f"{agent_type} {query}"

        # Search for relevant knowledge
        results = self.kb.semantic_search_scenes(
            agent_query, limit=limit, similarity_threshold=0.3
        )

        # Filter by agent relevance
        relevant_results = []
        for result in results:
            if self._is_relevant_to_agent(result, agent_type):
                relevant_results.append(result)

        return relevant_results

    def _is_relevant_to_agent(self, result: Dict[str, Any], agent_type: str) -> bool:
        """Check if a knowledge result is relevant to a specific agent"""

        scene_type = result.get("scene_type", "")
        parameters = result.get("parameters", {})
        subcategory = parameters.get("subcategory", "")

        # Agent-specific relevance rules
        if agent_type == "scene_composer":
            return any(x in scene_type for x in ["scene_type", "workflow", "documentation", "api_knowledge"]) or \
                   subcategory == "scene_composer"
        elif agent_type == "asset_generator":
            return any(x in scene_type for x in ["astronomy_asset", "factories", "classes", "api_knowledge"]) or \
                   subcategory == "asset_generator"
        elif agent_type == "terrain_engineer":
            return (any(x in scene_type for x in ["scene_type", "performance", "api_knowledge"]) and 
                   ("terrain" in str(parameters) or subcategory == "terrain_engineer"))
        elif agent_type == "render_controller":
            return (any(x in scene_type for x in ["performance", "workflow", "api_knowledge"]) and 
                   ("render" in str(parameters) or subcategory == "render_controller"))
        elif agent_type == "data_manager":
            return any(x in scene_type for x in ["workflow", "documentation", "api_knowledge"]) or \
                   subcategory == "data_manager"
        elif agent_type == "export_specialist":
            return (any(x in scene_type for x in ["workflow", "documentation", "api_knowledge"]) and 
                   ("export" in str(parameters) or subcategory == "export_specialist"))
        elif agent_type == "addon_manager":
            return subcategory == "addon_manager" or "addon" in str(parameters)

        return True  # Default to relevant


def main():
    """Main function to populate Infinigen agent knowledge"""

    # Initialize knowledge base
    kb = KnowledgeBase(db_path=Path("./infinigen_agent_knowledge.db"))

    # Create agent knowledge system
    agent_kb = InfinigenAgentKnowledge(kb)

    # Store all knowledge
    agent_kb.store_infinigen_knowledge()

    # Test queries for different agents
    logger.info("\n🔍 Testing agent-specific queries...")

    agents = [
        "scene_composer",
        "asset_generator",
        "terrain_engineer",
        "render_controller",
    ]

    for agent in agents:
        results = agent_kb.query_agent_knowledge(
            agent, "configuration parameters", limit=3
        )
        logger.info(f"{agent}: {len(results)} relevant results")
        for i, result in enumerate(results):
            logger.info(
                f"  {i+1}. {result['scene_type']} - {result['parameters'].get('subcategory', 'unknown')}"
            )

    # Close knowledge base
    kb.close()

    logger.info("\n✅ Infinigen Agent Knowledge populated successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
