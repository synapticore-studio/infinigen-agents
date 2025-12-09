# Data Manager Agent
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from config.model_factory import get_model

from deps.config_deps import InfinigenConfigDep
from deps.core_deps import SceneInfoManagerDep, TaskManagerDep
from tools.file_tools import FileManagerDep, LoggerDep

logger = logging.getLogger(__name__)


class DataManagerAgent(BaseModel):
    """Agent specialized in data management and job processing"""

    def __init__(self, **data):
        super().__init__(**data)

        # Agent configuration
        self.agent = Agent(
            get_model(),
            result_type=Dict[str, Any],
            system_prompt="""You are a specialized data management agent for Infinigen.
            
            Your responsibilities:
            - Create and manage data generation jobs
            - Monitor job progress and status
            - Handle distributed computing (Slurm)
            - Manage data storage and organization
            
            Job management features:
            - Create jobs with multiple scenes and tasks
            - Monitor job progress in real-time
            - Handle job failures and retries
            - Manage resource allocation
            
            Available tasks:
            - coarse: Basic scene composition
            - populate: Asset placement
            - fine_terrain: Detailed terrain generation
            - render: Scene rendering
            - ground_truth: GT annotation generation
            - export: Data export
            
            Always validate job parameters and provide detailed feedback on job status.
            Consider resource constraints when scheduling jobs.
            """,
        )

    def create_data_generation_job(
        self,
        job_name: str,
        output_folder: Path,
        scene_seeds: List[int],
        tasks: List[str],
        config: InfinigenConfigDep,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
        job_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create a data generation job with AI assistance"""
        try:
            # Validate job parameters
            if not scene_seeds:
                return {"success": False, "error": "No scene seeds provided"}

            if not tasks:
                return {"success": False, "error": "No tasks specified"}

            # Default job configuration
            default_config = {
                "max_parallel_jobs": config.max_workers,
                "memory_per_job": config.memory_limit,
                "time_limit": "2:00:00",
                "gpu_required": config.gpu_enabled,
            }

            if job_config:
                default_config.update(job_config)

            # Create job metadata
            job_metadata = {
                "job_name": job_name,
                "output_folder": str(output_folder),
                "scene_seeds": scene_seeds,
                "tasks": tasks,
                "config": default_config,
                "status": "created",
            }

            # Save job metadata
            job_file = output_folder / f"{job_name}_job.json"
            success = file_manager.save_json(job_metadata, job_file)

            if success:
                logger_tool.info(
                    f"Successfully created job: {job_name} with {len(scene_seeds)} scenes"
                )
                return {
                    "success": True,
                    "job_name": job_name,
                    "job_file": str(job_file),
                    "job_config": default_config,
                }
            else:
                return {"success": False, "error": "Failed to save job metadata"}

        except Exception as e:
            logger.error(f"Job creation failed: {e}")
            return {"success": False, "error": str(e)}

    def monitor_job_progress(
        self,
        job_name: str,
        file_manager: FileManagerDep,
        logger_tool: LoggerDep,
        detailed: bool = False,
    ) -> Dict[str, Any]:
        """Monitor job progress with AI assistance"""
        try:
            # This would typically read from job system
            # For now, return mock status
            status = {
                "job_name": job_name,
                "status": "running",
                "progress": 0.5,
                "completed_tasks": 5,
                "total_tasks": 10,
            }

            logger_tool.info(
                f"Job {job_name} status: {status.get('status', 'unknown')}"
            )
            return {
                "success": True,
                "job_name": job_name,
                "status": status,
                "detailed_info": {"mock": "data"} if detailed else None,
            }

        except Exception as e:
            logger.error(f"Job monitoring failed: {e}")
            return {"success": False, "error": str(e)}

    def get_job_recommendations(
        self,
        scene_count: int,
        complexity: str = "medium",
        config: InfinigenConfigDep = None,
    ) -> Dict[str, Any]:
        """Get AI-powered job configuration recommendations"""
        try:
            recommendations = {
                "scene_count": scene_count,
                "complexity": complexity,
                "recommended_config": self._get_job_config_recommendations(
                    scene_count, complexity
                ),
                "resource_estimates": self._get_resource_estimates(
                    scene_count, complexity
                ),
                "optimization_tips": self._get_optimization_tips(complexity),
            }

            return {"success": True, "recommendations": recommendations}

        except Exception as e:
            logger.error(f"Failed to get job recommendations: {e}")
            return {"success": False, "error": str(e)}

    def _get_job_config_recommendations(
        self, scene_count: int, complexity: str
    ) -> Dict[str, Any]:
        """Get job configuration recommendations"""
        config_map = {
            "low": {
                "max_parallel_jobs": min(scene_count, 8),
                "memory_per_job": "4GB",
                "time_limit": "1:00:00",
                "gpu_required": False,
            },
            "medium": {
                "max_parallel_jobs": min(scene_count, 4),
                "memory_per_job": "8GB",
                "time_limit": "2:00:00",
                "gpu_required": True,
            },
            "high": {
                "max_parallel_jobs": min(scene_count, 2),
                "memory_per_job": "16GB",
                "time_limit": "4:00:00",
                "gpu_required": True,
            },
        }
        return config_map.get(complexity, config_map["medium"])

    def _get_resource_estimates(
        self, scene_count: int, complexity: str
    ) -> Dict[str, Any]:
        """Get resource usage estimates"""
        estimates = {
            "low": {
                "total_memory": f"{scene_count * 4}GB",
                "estimated_time": f"{scene_count * 0.5} hours",
            },
            "medium": {
                "total_memory": f"{scene_count * 8}GB",
                "estimated_time": f"{scene_count * 1.0} hours",
            },
            "high": {
                "total_memory": f"{scene_count * 16}GB",
                "estimated_time": f"{scene_count * 2.0} hours",
            },
        }
        return estimates.get(complexity, estimates["medium"])

    def _get_optimization_tips(self, complexity: str) -> List[str]:
        """Get optimization tips"""
        tips = {
            "low": [
                "Use lower resolution renders",
                "Disable unnecessary features",
                "Use CPU rendering",
            ],
            "medium": [
                "Balance quality vs performance",
                "Use GPU acceleration",
                "Optimize asset complexity",
            ],
            "high": [
                "Use high-quality settings",
                "Enable all features",
                "Use multiple GPUs if available",
            ],
        }
        return tips.get(complexity, tips["medium"])
