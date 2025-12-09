#!/usr/bin/env python3
"""
DuckDB Spatial Manager
Manages terrain data using DuckDB with spatial extensions
"""

import logging
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np

logger = logging.getLogger(__name__)


class DuckDBSpatialManager:
    """Manages terrain data using DuckDB with spatial extensions"""

    def __init__(self, db_path: str = "terrain_data.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self._init_database()

    def _init_database(self):
        """Initialize DuckDB database with spatial extensions"""
        try:
            self.connection = duckdb.connect(self.db_path)
            
            # Enable spatial extension
            self.connection.execute("INSTALL spatial;")
            self.connection.execute("LOAD spatial;")
            
            # Create terrain data table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS terrain_data (
                    id INTEGER PRIMARY KEY,
                    terrain_type VARCHAR,
                    seed INTEGER,
                    resolution INTEGER,
                    bounds_x_min DOUBLE,
                    bounds_x_max DOUBLE,
                    bounds_y_min DOUBLE,
                    bounds_y_max DOUBLE,
                    bounds_z_min DOUBLE,
                    bounds_z_max DOUBLE,
                    height_map BLOB,
                    normal_map BLOB,
                    displacement_map BLOB,
                    roughness_map BLOB,
                    ao_map BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.logger.info("✅ DuckDB spatial database initialized")

        except Exception as e:
            self.logger.error(f"Error initializing DuckDB database: {e}")

    def store_terrain_data(self, terrain_obj: Any, height_map: np.ndarray, 
                          terrain_type: str = "mountain", seed: int = 42, 
                          resolution: int = 256, bounds: tuple = None) -> bool:
        """Store terrain data in DuckDB"""
        try:
            if not self.connection:
                self.logger.error("Database connection not initialized")
                return False

            # Get terrain bounds
            if bounds is None:
                bounds = self._get_terrain_bounds(terrain_obj)

            # Convert maps to binary
            height_map_blob = height_map.tobytes()
            
            # Get next ID
            result = self.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM terrain_data").fetchone()
            next_id = result[0] if result else 1

            # Insert terrain data
            self.connection.execute("""
                INSERT INTO terrain_data 
                (id, terrain_type, seed, resolution, bounds_x_min, bounds_x_max, 
                 bounds_y_min, bounds_y_max, bounds_z_min, bounds_z_max, height_map)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (next_id, terrain_type, seed, resolution, bounds[0], bounds[1], 
                  bounds[2], bounds[3], bounds[4], bounds[5], height_map_blob))

            self.logger.info(f"✅ Terrain data stored: {terrain_type}_{seed}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing terrain data: {e}")
            return False

    def query_terrain_data(self, terrain_type: str = None, seed: int = None) -> List[Dict]:
        """Query terrain data from DuckDB"""
        try:
            if not self.connection:
                self.logger.error("Database connection not initialized")
                return []

            query = "SELECT * FROM terrain_data WHERE 1=1"
            params = []
            
            if terrain_type:
                query += " AND terrain_type = ?"
                params.append(terrain_type)
            
            if seed:
                query += " AND seed = ?"
                params.append(seed)
            
            query += " ORDER BY created_at DESC"
            
            result = self.connection.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [desc[0] for desc in self.connection.description]
            terrain_data = [dict(zip(columns, row)) for row in result]
            
            self.logger.info(f"✅ Queried {len(terrain_data)} terrain records")
            return terrain_data

        except Exception as e:
            self.logger.error(f"Error querying terrain data: {e}")
            return []

    def get_terrain_by_id(self, terrain_id: int) -> Optional[Dict]:
        """Get specific terrain data by ID"""
        try:
            if not self.connection:
                return None

            result = self.connection.execute(
                "SELECT * FROM terrain_data WHERE id = ?", (terrain_id,)
            ).fetchone()
            
            if result:
                columns = [desc[0] for desc in self.connection.description]
                return dict(zip(columns, result))
            
            return None

        except Exception as e:
            self.logger.error(f"Error getting terrain by ID: {e}")
            return None

    def _get_terrain_bounds(self, terrain_obj: Any) -> tuple:
        """Get terrain object bounds"""
        try:
            if hasattr(terrain_obj, 'bound_box'):
                # Blender object
                bbox = terrain_obj.bound_box
                min_coords = [min(coord[i] for coord in bbox) for i in range(3)]
                max_coords = [max(coord[i] for coord in bbox) for i in range(3)]
                return (min_coords[0], max_coords[0], min_coords[1], max_coords[1], 
                       min_coords[2], max_coords[2])
            else:
                # Default bounds
                return (-50, 50, -50, 50, 0, 100)

        except Exception as e:
            self.logger.error(f"Error getting terrain bounds: {e}")
            return (-50, 50, -50, 50, 0, 100)

    def close(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                self.logger.info("✅ DuckDB connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
