#!/usr/bin/env python3
"""
Script to fill the Knowledge Base with real Infinigen data
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from deps.knowledge_deps import KnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_scene_folder(scene_folder: Path) -> Dict[str, Any]:
    """Analyze a scene folder and extract metadata"""
    scene_info = {
        "scene_type": "unknown",
        "scene_seed": 42,
        "success": True,
        "parameters": {},
        "performance_metrics": {},
        "generated_assets": [],
        "error_messages": None,
        "file_size": 0,
        "num_objects": 0,
        "complexity": "low",
    }

    try:
        # Determine scene type from folder name
        folder_name = scene_folder.name.lower()
        if "astronomy" in folder_name:
            scene_info["scene_type"] = "astronomy"
        elif "forest" in folder_name:
            scene_info["scene_type"] = "forest"
        elif "indoor" in folder_name:
            scene_info["scene_type"] = "indoor"
        elif "nature" in folder_name:
            scene_info["scene_type"] = "nature"

        # Check for .blend file
        blend_files = list(scene_folder.glob("*.blend"))
        if blend_files:
            scene_info["file_size"] = blend_files[0].stat().st_size
            scene_info["success"] = True
        else:
            scene_info["success"] = False
            scene_info["error_messages"] = "No .blend file found"

        # Analyze mesh data
        mesh_json_files = list(scene_folder.glob("**/saved_mesh.json"))
        if mesh_json_files:
            try:
                with open(mesh_json_files[0], "r") as f:
                    mesh_data = json.load(f)

                scene_info["num_objects"] = len(mesh_data)
                scene_info["generated_assets"] = [
                    obj.get("object_name", f"object_{i}")
                    for i, obj in enumerate(mesh_data)
                ]

                # Calculate complexity based on object count and vertices
                total_verts = sum(obj.get("num_verts", 0) for obj in mesh_data)
                if total_verts > 10000:
                    scene_info["complexity"] = "high"
                elif total_verts > 1000:
                    scene_info["complexity"] = "medium"
                else:
                    scene_info["complexity"] = "low"

                # Extract performance metrics from mesh data
                scene_info["performance_metrics"] = {
                    "total_vertices": total_verts,
                    "total_faces": sum(obj.get("num_faces", 0) for obj in mesh_data),
                    "object_count": len(mesh_data),
                    "file_size_mb": scene_info["file_size"] / (1024 * 1024),
                }

            except Exception as e:
                logger.warning(
                    f"Could not parse mesh data from {mesh_json_files[0]}: {e}"
                )

        # Extract parameters from folder structure
        scene_info["parameters"] = {
            "complexity": scene_info["complexity"],
            "output_type": "blend",
            "has_mesh_data": len(mesh_json_files) > 0,
            "folder_name": folder_name,
        }

    except Exception as e:
        logger.error(f"Error analyzing scene folder {scene_folder}: {e}")
        scene_info["success"] = False
        scene_info["error_messages"] = str(e)

    return scene_info


def fill_knowledge_base_with_infinigen_data():
    """Fill the knowledge base with real Infinigen output data"""

    # Initialize knowledge base
    kb = KnowledgeBase(db_path=Path("./infinigen_knowledge.db"))

    # Define output folders to analyze
    output_folders = [
        "astronomy_complete_output",
        "astronomy_generated_output",
        "astronomy_film_output",
        "simple_astronomy_output",
        "test_output",
        "test_output2",
    ]

    total_scenes = 0
    successful_scenes = 0

    for folder_name in output_folders:
        folder_path = Path(folder_name)
        if not folder_path.exists():
            logger.warning(f"Folder {folder_name} does not exist, skipping")
            continue

        logger.info(f"Analyzing folder: {folder_name}")

        # Analyze the main scene folder
        scene_info = analyze_scene_folder(folder_path)
        scene_info["scene_seed"] = (
            hash(folder_name) % 10000
        )  # Generate seed from folder name

        # Store in knowledge base
        try:
            scene_id = kb.store_scene_knowledge(
                scene_type=scene_info["scene_type"],
                scene_seed=scene_info["scene_seed"],
                success=scene_info["success"],
                parameters=scene_info["parameters"],
                performance_metrics=scene_info["performance_metrics"],
                generated_assets=scene_info["generated_assets"],
                error_messages=scene_info["error_messages"],
            )

            total_scenes += 1
            if scene_info["success"]:
                successful_scenes += 1

            logger.info(
                f"✅ Stored scene {scene_id}: {scene_info['scene_type']} "
                f"(success: {scene_info['success']}, objects: {scene_info['num_objects']})"
            )

        except Exception as e:
            logger.error(f"Failed to store scene from {folder_name}: {e}")

    # Add some synthetic knowledge for different scene types
    synthetic_scenes = [
        {
            "scene_type": "forest",
            "scene_seed": 1234,
            "success": True,
            "parameters": {
                "complexity": "high",
                "tree_density": "dense",
                "season": "autumn",
            },
            "performance_metrics": {
                "execution_time": 45.2,
                "memory_usage": 1024,
                "total_vertices": 50000,
            },
            "generated_assets": [
                "oak_tree_001",
                "pine_tree_002",
                "rock_003",
                "grass_patch_004",
            ],
            "error_messages": None,
        },
        {
            "scene_type": "indoor",
            "scene_seed": 5678,
            "success": True,
            "parameters": {
                "complexity": "medium",
                "room_type": "kitchen",
                "furniture_density": "normal",
            },
            "performance_metrics": {
                "execution_time": 23.1,
                "memory_usage": 512,
                "total_vertices": 25000,
            },
            "generated_assets": [
                "kitchen_cabinet_001",
                "stove_002",
                "sink_003",
                "table_004",
            ],
            "error_messages": None,
        },
        {
            "scene_type": "desert",
            "scene_seed": 9999,
            "success": False,
            "parameters": {"complexity": "high", "sand_dunes": "many", "oasis": True},
            "performance_metrics": {
                "execution_time": 60.0,
                "memory_usage": 2048,
                "total_vertices": 75000,
            },
            "generated_assets": [],
            "error_messages": "Memory limit exceeded during terrain generation",
        },
    ]

    for scene_data in synthetic_scenes:
        try:
            scene_id = kb.store_scene_knowledge(**scene_data)
            total_scenes += 1
            if scene_data["success"]:
                successful_scenes += 1
            logger.info(
                f"✅ Stored synthetic scene {scene_id}: {scene_data['scene_type']}"
            )
        except Exception as e:
            logger.error(f"Failed to store synthetic scene: {e}")

    # Store asset knowledge
    asset_examples = [
        {
            "asset_type": "tree",
            "asset_category": "vegetation",
            "complexity": "high",
            "success": True,
            "parameters": {"species": "oak", "height": 15.0, "season": "autumn"},
            "performance_metrics": {
                "generation_time": 5.2,
                "vertex_count": 8000,
                "memory_usage": 64,
            },
            "quality_score": 0.85,
            "error_messages": None,
        },
        {
            "asset_type": "rock",
            "asset_category": "terrain",
            "complexity": "medium",
            "success": True,
            "parameters": {"size": "large", "texture": "granite", "weathering": "high"},
            "performance_metrics": {
                "generation_time": 2.1,
                "vertex_count": 2000,
                "memory_usage": 32,
            },
            "quality_score": 0.92,
            "error_messages": None,
        },
    ]

    for asset_data in asset_examples:
        try:
            asset_id = kb.store_asset_knowledge(**asset_data)
            logger.info(f"✅ Stored asset {asset_id}: {asset_data['asset_type']}")
        except Exception as e:
            logger.error(f"Failed to store asset: {e}")

    # Store terrain knowledge
    terrain_examples = [
        {
            "terrain_type": "mountain",
            "detail_level": "high",
            "success": True,
            "parameters": {"height": 1000.0, "roughness": 0.7, "vegetation": True},
            "performance_metrics": {
                "generation_time": 30.5,
                "vertex_count": 100000,
                "memory_usage": 512,
            },
            "quality_score": 0.88,
            "error_messages": None,
        }
    ]

    for terrain_data in terrain_examples:
        try:
            terrain_id = kb.store_terrain_knowledge(**terrain_data)
            logger.info(
                f"✅ Stored terrain {terrain_id}: {terrain_data['terrain_type']}"
            )
        except Exception as e:
            logger.error(f"Failed to store terrain: {e}")

    # Store render knowledge
    render_examples = [
        {
            "render_engine": "BLENDER_EEVEE_NEXT",
            "quality_setting": "high",
            "success": True,
            "parameters": {"samples": 128, "resolution": "4K", "denoising": True},
            "performance_metrics": {
                "render_time": 120.5,
                "memory_usage": 2048,
                "gpu_usage": 0.85,
            },
            "quality_score": 0.95,
            "error_messages": None,
        }
    ]

    for render_data in render_examples:
        try:
            render_id = kb.store_render_knowledge(**render_data)
            logger.info(f"✅ Stored render {render_id}: {render_data['render_engine']}")
        except Exception as e:
            logger.error(f"Failed to store render: {e}")

    # Test semantic search
    logger.info("\n🔍 Testing semantic search...")

    search_queries = [
        "astronomy scene generation",
        "forest with trees",
        "indoor kitchen scene",
        "high complexity terrain",
        "failed generation",
    ]

    for query in search_queries:
        results = kb.semantic_search_scenes(query, limit=3, similarity_threshold=0.1)
        logger.info(f"Query '{query}': {len(results)} results")
        for i, result in enumerate(results):
            logger.info(
                f"  {i+1}. {result['scene_type']} (similarity: {result['similarity']:.3f})"
            )

    # Test similar cases
    logger.info("\n🔍 Testing similar cases...")
    similar_cases = kb.get_similar_successful_cases(
        "astronomy", {"complexity": "high"}, limit=3
    )
    logger.info(f"Similar cases for astronomy: {len(similar_cases)} results")

    # Close knowledge base
    kb.close()

    logger.info(f"\n✅ Knowledge Base filled successfully!")
    logger.info(f"Total scenes: {total_scenes}")
    logger.info(f"Successful scenes: {successful_scenes}")
    logger.info(
        f"Success rate: {successful_scenes/total_scenes*100:.1f}%"
        if total_scenes > 0
        else "No scenes processed"
    )


if __name__ == "__main__":
    fill_knowledge_base_with_infinigen_data()
