#!/usr/bin/env python3
"""
Infinigen AI System
Coordinated AI agents for procedural generation with Infinigen

NOTE: This file needs to be updated to work with the new agent pattern.
Agents are now pydantic-ai Agent instances, not classes.
They should be used with agent.run() or agent.run_sync() with dependencies.

This system provides specialized AI agents for different aspects of Infinigen:
- Scene Composition
- Asset Generation
- Terrain Engineering
- Render Control
- Data Management
- Export Specialization

Each agent has minimal dependencies and focused tools for maximum efficiency.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agents import (
    AssetGeneratorAgent,
    DataManagerAgent,
    ExportSpecialistAgent,
    RenderControllerAgent,
    SceneComposerAgent,
    TerrainEngineerAgent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class InfinigenAISystem(BaseModel):
    """Main coordinator for Infinigen AI agents"""

    # Initialize all agents
    scene_composer: SceneComposerAgent = SceneComposerAgent()
    asset_generator: AssetGeneratorAgent = AssetGeneratorAgent()
    terrain_engineer: TerrainEngineerAgent = TerrainEngineerAgent()
    render_controller: RenderControllerAgent = RenderControllerAgent()
    data_manager: DataManagerAgent = DataManagerAgent()
    export_specialist: ExportSpecialistAgent = ExportSpecialistAgent()

    def create_complete_scene(
        self,
        output_folder: Path,
        scene_seed: int,
        scene_type: str = "forest",
        complexity: str = "medium",
        include_terrain: bool = True,
        include_rendering: bool = True,
        include_export: bool = False,
    ) -> Dict[str, Any]:
        """Create a complete scene using all relevant agents"""
        try:
            logger.info(f"Creating complete {scene_type} scene with seed {scene_seed}")

            # Step 1: Compose the scene
            scene_result = self.scene_composer.compose_nature_scene(
                output_folder=output_folder,
                scene_seed=scene_seed,
                scene_type=scene_type,
            )

            if not scene_result["success"]:
                return scene_result

            # Step 2: Generate terrain if requested
            terrain_result = None
            if include_terrain:
                terrain_result = self.terrain_engineer.generate_terrain(
                    output_folder=output_folder,
                    scene_seed=scene_seed,
                    terrain_type=scene_type,
                    detail_level=complexity,
                )

                if not terrain_result["success"]:
                    logger.warning(
                        f"Terrain generation failed: {terrain_result.get('error')}"
                    )

            # Step 3: Render the scene if requested
            render_result = None
            if include_rendering:
                render_result = self.render_controller.render_scene(
                    scene_folder=output_folder, output_folder=output_folder / "renders"
                )

                if not render_result["success"]:
                    logger.warning(f"Rendering failed: {render_result.get('error')}")

            # Step 4: Export if requested
            export_result = None
            if include_export:
                blend_file = output_folder / "scene.blend"
                if blend_file.exists():
                    export_result = self.export_specialist.export_scene_data(
                        input_blend_file=blend_file,
                        output_folder=output_folder / "exports",
                    )

                    if not export_result["success"]:
                        logger.warning(f"Export failed: {export_result.get('error')}")

            return {
                "success": True,
                "scene_type": scene_type,
                "seed": scene_seed,
                "complexity": complexity,
                "scene_result": scene_result,
                "terrain_result": terrain_result,
                "render_result": render_result,
                "export_result": export_result,
                "output_folder": str(output_folder),
            }

        except Exception as e:
            logger.error(f"Complete scene creation failed: {e}")
            return {"success": False, "error": str(e)}

    def generate_asset_library(
        self,
        output_folder: Path,
        asset_types: List[str],
        count_per_type: int = 5,
        complexity: str = "medium",
    ) -> Dict[str, Any]:
        """Generate a library of assets using the asset generator"""
        try:
            logger.info(f"Generating asset library with {len(asset_types)} types")

            results = {}
            total_assets = 0

            for asset_type in asset_types:
                asset_results = []

                for i in range(count_per_type):
                    seed = 42 + i
                    asset_folder = output_folder / asset_type / f"asset_{i:03d}"
                    asset_folder.mkdir(parents=True, exist_ok=True)

                    if asset_type in ["carnivore", "herbivore", "bird", "fish"]:
                        result = self.asset_generator.generate_creature_asset(
                            creature_type=asset_type,
                            output_path=asset_folder,
                            seed=seed,
                            complexity=complexity,
                        )
                    elif asset_type in ["pine", "oak", "palm", "bamboo"]:
                        result = self.asset_generator.generate_tree_asset(
                            tree_type=asset_type,
                            output_path=asset_folder,
                            seed=seed,
                            complexity=complexity,
                        )
                    elif asset_type in ["ground", "water", "rock", "snow", "sand"]:
                        result = self.asset_generator.generate_material_asset(
                            material_type=asset_type, output_path=asset_folder
                        )
                    else:
                        result = {
                            "success": False,
                            "error": f"Unknown asset type: {asset_type}",
                        }

                    asset_results.append(result)
                    if result["success"]:
                        total_assets += 1

                results[asset_type] = asset_results

            return {
                "success": True,
                "asset_types": asset_types,
                "count_per_type": count_per_type,
                "total_assets": total_assets,
                "results": results,
                "output_folder": str(output_folder),
            }

        except Exception as e:
            logger.error(f"Asset library generation failed: {e}")
            return {"success": False, "error": str(e)}

    def create_data_generation_pipeline(
        self,
        job_name: str,
        output_folder: Path,
        scene_configs: List[Dict[str, Any]],
        tasks: List[str] = None,
    ) -> Dict[str, Any]:
        """Create a data generation pipeline using the data manager"""
        try:
            if tasks is None:
                tasks = ["coarse", "populate", "fine_terrain", "render", "ground_truth"]

            # Extract scene seeds from configs
            scene_seeds = [config["seed"] for config in scene_configs]

            # Create the job
            job_result = self.data_manager.create_data_generation_job(
                job_name=job_name,
                output_folder=output_folder,
                scene_seeds=scene_seeds,
                tasks=tasks,
            )

            if not job_result["success"]:
                return job_result

            # Get job recommendations
            recommendations = self.data_manager.get_job_recommendations(
                scene_count=len(scene_seeds), complexity="medium"
            )

            return {
                "success": True,
                "job_name": job_name,
                "scene_count": len(scene_seeds),
                "tasks": tasks,
                "job_result": job_result,
                "recommendations": recommendations,
                "output_folder": str(output_folder),
            }

        except Exception as e:
            logger.error(f"Data generation pipeline creation failed: {e}")
            return {"success": False, "error": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents and their dependencies"""
        try:
            status = {
                "scene_composer": self.scene_composer.scene_tools.validate_scene_folder(
                    Path(".")
                ),
                "asset_generator": self.asset_generator.asset_tools.list_available_assets(),
                "terrain_engineer": self.terrain_engineer.validate_terrain_setup(),
                "render_controller": self.render_controller.validate_render_setup(),
                "data_manager": self.data_manager.validate_data_setup(),
                "export_specialist": self.export_specialist.validate_export_setup(),
            }

            return {"success": True, "status": status}

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"success": False, "error": str(e)}

    def get_agent_recommendations(
        self, task_type: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommendations from all relevant agents"""
        try:
            recommendations = {}

            if task_type in ["scene", "nature", "indoor"]:
                recommendations["scene_composer"] = (
                    self.scene_composer.get_scene_recommendations(
                        scene_type=requirements.get("scene_type", "forest"),
                        complexity=requirements.get("complexity", "medium"),
                    )
                )

            if task_type in ["assets", "creatures", "trees", "materials"]:
                recommendations["asset_generator"] = (
                    self.asset_generator.get_asset_recommendations(
                        scene_type=requirements.get("scene_type", "forest"),
                        asset_category=requirements.get("asset_category", "trees"),
                    )
                )

            if task_type in ["terrain", "landscape"]:
                recommendations["terrain_engineer"] = (
                    self.terrain_engineer.get_terrain_recommendations(
                        scene_type=requirements.get("scene_type", "forest"),
                        performance_requirements=requirements.get(
                            "performance", "medium"
                        ),
                    )
                )

            if task_type in ["rendering", "visualization"]:
                recommendations["render_controller"] = (
                    self.render_controller.get_render_recommendations(
                        scene_type=requirements.get("scene_type", "forest"),
                        quality_requirements=requirements.get("quality", "medium"),
                    )
                )

            if task_type in ["data", "jobs", "pipeline"]:
                recommendations["data_manager"] = (
                    self.data_manager.get_job_recommendations(
                        scene_count=requirements.get("scene_count", 10),
                        complexity=requirements.get("complexity", "medium"),
                    )
                )

            if task_type in ["export", "conversion"]:
                recommendations["export_specialist"] = (
                    self.export_specialist.get_export_recommendations(
                        target_application=requirements.get(
                            "target_application", "research"
                        ),
                        data_type=requirements.get("data_type", "scene"),
                    )
                )

            return {"success": True, "recommendations": recommendations}

        except Exception as e:
            logger.error(f"Failed to get agent recommendations: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main function for testing the AI system"""
    # Initialize the system
    system = InfinigenAISystem()

    # Test system status
    status = system.get_system_status()
    print("System Status:", status)

    # Test recommendations
    recommendations = system.get_agent_recommendations(
        task_type="scene", requirements={"scene_type": "forest", "complexity": "medium"}
    )
    print("Recommendations:", recommendations)


if __name__ == "__main__":
    main()
    main()
