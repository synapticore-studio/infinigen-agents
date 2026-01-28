# Knowledge Management Dependencies - DuckDB + VSS + AST UDFs
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np

# Simple dependency injection without pydantic-ai Depends
from sentence_transformers import SentenceTransformer

from infinigen.core.util.organization import Task


@dataclass
class KnowledgeBase:
    """Intelligent knowledge base using DuckDB + VSS + AST UDFs"""

    def __init__(self, db_path: Path = Path("./knowledge.db")):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger = logging.getLogger(__name__)

        # Initialize knowledge tables
        self._init_knowledge_tables()

    def _init_knowledge_tables(self):
        """Initialize DuckDB tables for knowledge storage"""

        # Install and load VSS extension
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            self.logger.info("VSS extension loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load VSS extension: {e}")

        # Scene Generation Knowledge
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_knowledge (
                id INTEGER PRIMARY KEY,
                scene_type VARCHAR,
                scene_seed INTEGER,
                success BOOLEAN,
                parameters JSON,
                performance_metrics JSON,
                generated_assets JSON,
                error_messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]  -- VSS embedding
            )
        """
        )

        # Asset Generation Knowledge
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_knowledge (
                id INTEGER PRIMARY KEY,
                asset_type VARCHAR,
                asset_category VARCHAR,
                complexity VARCHAR,
                success BOOLEAN,
                parameters JSON,
                performance_metrics JSON,
                quality_score FLOAT,
                error_messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]
            )
        """
        )

        # Terrain Generation Knowledge
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terrain_knowledge (
                id INTEGER PRIMARY KEY,
                terrain_type VARCHAR,
                detail_level VARCHAR,
                success BOOLEAN,
                parameters JSON,
                performance_metrics JSON,
                quality_score FLOAT,
                error_messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]
            )
        """
        )

        # Render Knowledge
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS render_knowledge (
                id INTEGER PRIMARY KEY,
                render_engine VARCHAR,
                quality_setting VARCHAR,
                success BOOLEAN,
                parameters JSON,
                performance_metrics JSON,
                quality_score FLOAT,
                error_messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384]
            )
        """
        )

        # Agent Performance Knowledge
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY,
                agent_name VARCHAR,
                task_type VARCHAR,
                success_rate FLOAT,
                avg_execution_time FLOAT,
                common_errors JSON,
                best_practices JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # VSS indexes will be created after data insertion for better performance

    def store_scene_knowledge(
        self,
        scene_type: str,
        scene_seed: int,
        success: bool,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        generated_assets: List[str],
        error_messages: Optional[str] = None,
    ) -> int:
        """Store scene generation knowledge"""

        # Create embedding from parameters and context
        context_text = (
            f"{scene_type} {json.dumps(parameters)} {json.dumps(performance_metrics)}"
        )
        embedding = self.embedding_model.encode(context_text).tolist()

        # Get next ID
        max_id_result = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM scene_knowledge"
        ).fetchone()
        next_id = (max_id_result[0] if max_id_result else 0) + 1

        # Insert into database
        result = self.conn.execute(
            """
            INSERT INTO scene_knowledge 
            (id, scene_type, scene_seed, success, parameters, performance_metrics, 
             generated_assets, error_messages, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                next_id,
                scene_type,
                scene_seed,
                success,
                json.dumps(parameters),
                json.dumps(performance_metrics),
                json.dumps(generated_assets),
                error_messages,
                embedding,
            ],
        )

        return next_id

    def store_asset_knowledge(
        self,
        asset_type: str,
        asset_category: str,
        complexity: str,
        success: bool,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        quality_score: float,
        error_messages: Optional[str] = None,
    ) -> int:
        """Store asset generation knowledge"""

        context_text = (
            f"{asset_type} {asset_category} {complexity} {json.dumps(parameters)}"
        )
        embedding = self.embedding_model.encode(context_text).tolist()

        result = self.conn.execute(
            """
            INSERT INTO asset_knowledge 
            (asset_type, asset_category, complexity, success, parameters, 
             performance_metrics, quality_score, error_messages, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """,
            [
                asset_type,
                asset_category,
                complexity,
                success,
                json.dumps(parameters),
                json.dumps(performance_metrics),
                quality_score,
                error_messages,
                embedding,
            ],
        ).fetchone()

        return result[0] if result else None

    def semantic_search_scenes(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Semantic search for similar scene generations"""

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.conn.execute(
            """
            SELECT scene_type, scene_seed, success, parameters, 
                   performance_metrics, generated_assets, error_messages,
                   array_cosine_similarity(embedding, ?::FLOAT[384]) as similarity
            FROM scene_knowledge
            WHERE array_cosine_similarity(embedding, ?::FLOAT[384]) > ?
            ORDER BY similarity DESC
            LIMIT ?
        """,
            [query_embedding, query_embedding, similarity_threshold, limit],
        ).fetchall()

        return [
            {
                "scene_type": row[0],
                "scene_seed": row[1],
                "success": row[2],
                "parameters": json.loads(row[3]),
                "performance_metrics": json.loads(row[4]),
                "generated_assets": json.loads(row[5]),
                "error_messages": row[6],
                "similarity": row[7],
            }
            for row in results
        ]

    def get_best_practices(self, agent_name: str, task_type: str) -> Dict[str, Any]:
        """Get best practices for specific agent and task"""

        result = self.conn.execute(
            """
            SELECT best_practices, success_rate, common_errors
            FROM agent_performance
            WHERE agent_name = ? AND task_type = ?
            ORDER BY updated_at DESC
            LIMIT 1
        """,
            [agent_name, task_type],
        ).fetchone()

        if result:
            return {
                "best_practices": json.loads(result[0]) if result[0] else {},
                "success_rate": result[1],
                "common_errors": json.loads(result[2]) if result[2] else [],
            }

        return {"best_practices": {}, "success_rate": 0.0, "common_errors": []}

    def update_agent_performance(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        execution_time: float,
        error_message: Optional[str] = None,
    ):
        """Update agent performance metrics"""

        # Get current performance
        current = self.conn.execute(
            """
            SELECT success_rate, avg_execution_time, common_errors
            FROM agent_performance
            WHERE agent_name = ? AND task_type = ?
        """,
            [agent_name, task_type],
        ).fetchone()

        if current:
            # Update existing record
            new_success_rate = (current[0] + (1.0 if success else 0.0)) / 2
            new_avg_time = (current[1] + execution_time) / 2

            common_errors = json.loads(current[2]) if current[2] else []
            if error_message and error_message not in common_errors:
                common_errors.append(error_message)

            self.conn.execute(
                """
                UPDATE agent_performance
                SET success_rate = ?, avg_execution_time = ?, 
                    common_errors = ?, updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND task_type = ?
            """,
                [
                    new_success_rate,
                    new_avg_time,
                    json.dumps(common_errors),
                    agent_name,
                    task_type,
                ],
            )
        else:
            # Create new record - get next ID
            max_id_result = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM agent_performance"
            ).fetchone()
            next_id = max_id_result[0] + 1 if max_id_result else 1

            common_errors = [error_message] if error_message else []
            self.conn.execute(
                """
                INSERT INTO agent_performance
                (id, agent_name, task_type, success_rate, avg_execution_time, common_errors)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    next_id,
                    agent_name,
                    task_type,
                    1.0 if success else 0.0,
                    execution_time,
                    json.dumps(common_errors),
                ],
            )

    def get_similar_successful_cases(
        self, scene_type: str, parameters: Dict[str, Any], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar successful cases for learning"""

        context_text = f"{scene_type} {json.dumps(parameters)}"
        query_embedding = self.embedding_model.encode(context_text).tolist()

        results = self.conn.execute(
            """
            SELECT scene_type, scene_seed, parameters, performance_metrics,
                   array_cosine_similarity(embedding, ?::FLOAT[384]) as similarity
            FROM scene_knowledge
            WHERE success = true AND array_cosine_similarity(embedding, ?::FLOAT[384]) > 0.6
            ORDER BY similarity DESC
            LIMIT ?
        """,
            [query_embedding, query_embedding, limit],
        ).fetchall()

        return [
            {
                "scene_type": row[0],
                "scene_seed": row[1],
                "parameters": json.loads(row[2]),
                "performance_metrics": json.loads(row[3]),
                "similarity": row[4],
            }
            for row in results
        ]

    def close(self):
        """Close database connection"""
        self.conn.close()


# Simple dependency injection
KnowledgeBaseDep = KnowledgeBase
