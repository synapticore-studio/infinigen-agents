#!/usr/bin/env python3
"""
OpenStreetMap Integration für Infinigen
Moderne Stadt- und Straßen-Generierung mit OSM-Daten
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
import numpy as np
import overpy
import requests

logger = logging.getLogger(__name__)


class OSMDataFetcher:
    """Lädt OpenStreetMap-Daten für ein Gebiet"""

    def __init__(self):
        self.api = overpy.Overpass()
        self.logger = logging.getLogger(__name__)

    def fetch_area_data(
        self,
        bbox: Tuple[float, float, float, float],  # min_lat, min_lon, max_lat, max_lon
        area_type: str = "urban",
    ) -> Dict[str, Any]:
        """Lade OSM-Daten für ein Gebiet"""

        min_lat, min_lon, max_lat, max_lon = bbox

        # OSM Query basierend auf Gebietstyp
        if area_type == "urban":
            query = f"""
            [out:json][timeout:25];
            (
              way["highway"~"^(primary|secondary|tertiary|residential|unclassified)$"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["building"~"^(yes|house|apartments|commercial|industrial)$"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["landuse"~"^(residential|commercial|industrial|park|forest)$"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            out geom;
            """
        elif area_type == "rural":
            query = f"""
            [out:json][timeout:25];
            (
              way["highway"~"^(primary|secondary|tertiary|track|path)$"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["natural"~"^(forest|wood|grassland|scrub)$"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["landuse"~"^(farmland|forest|meadow)$"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            out geom;
            """
        else:
            query = f"""
            [out:json][timeout:25];
            (
              way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["natural"]({min_lat},{min_lon},{max_lat},{max_lon});
              way["landuse"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            out geom;
            """

        try:
            result = self.api.query(query)

            # Daten strukturieren
            data = {
                "roads": [],
                "buildings": [],
                "landuse": [],
                "natural": [],
                "bbox": bbox,
                "area_type": area_type,
            }

            for way in result.ways:
                if "highway" in way.tags:
                    data["roads"].append(
                        {
                            "id": way.id,
                            "type": way.tags["highway"],
                            "name": way.tags.get("name", ""),
                            "nodes": [(node.lat, node.lon) for node in way.nodes],
                        }
                    )
                elif "building" in way.tags:
                    data["buildings"].append(
                        {
                            "id": way.id,
                            "type": way.tags["building"],
                            "height": way.tags.get("height", "3"),
                            "nodes": [(node.lat, node.lon) for node in way.nodes],
                        }
                    )
                elif "landuse" in way.tags:
                    data["landuse"].append(
                        {
                            "id": way.id,
                            "type": way.tags["landuse"],
                            "nodes": [(node.lat, node.lon) for node in way.nodes],
                        }
                    )
                elif "natural" in way.tags:
                    data["natural"].append(
                        {
                            "id": way.id,
                            "type": way.tags["natural"],
                            "nodes": [(node.lat, node.lon) for node in way.nodes],
                        }
                    )

            self.logger.info(
                f"✅ OSM-Daten geladen: {len(data['roads'])} Straßen, {len(data['buildings'])} Gebäude"
            )
            return data

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden der OSM-Daten: {e}")
            return {"error": str(e)}


class OSMToBlenderConverter:
    """Konvertiert OSM-Daten zu Blender-Objekten"""

    def __init__(self, scale_factor: float = 1000.0):
        self.scale_factor = scale_factor  # OSM-Koordinaten zu Blender-Metern
        self.logger = logging.getLogger(__name__)

    def latlon_to_blender(
        self, lat: float, lon: float, center_lat: float, center_lon: float
    ) -> Tuple[float, float]:
        """Konvertiere Lat/Lon zu Blender-Koordinaten"""
        # Vereinfachte Projektion (für kleine Gebiete ausreichend)
        x = (lon - center_lon) * self.scale_factor * np.cos(np.radians(center_lat))
        y = (lat - center_lat) * self.scale_factor
        return x, y

    def create_roads(
        self, osm_data: Dict[str, Any], center_lat: float, center_lon: float
    ) -> List[bpy.types.Object]:
        """Erstelle Straßen aus OSM-Daten"""
        roads = []

        for road in osm_data.get("roads", []):
            if len(road["nodes"]) < 2:
                continue

            # Straßen-Typ bestimmen
            road_type = road["type"]
            if road_type in ["primary", "secondary"]:
                width = 8.0
                height = 0.1
            elif road_type == "tertiary":
                width = 6.0
                height = 0.1
            elif road_type == "residential":
                width = 4.0
                height = 0.1
            else:
                width = 2.0
                height = 0.05

            # Straße erstellen
            road_obj = self._create_road_mesh(
                road["nodes"], width, height, center_lat, center_lon
            )
            if road_obj:
                road_obj.name = f"Road_{road_type}_{road['id']}"
                road_obj["road_type"] = road_type
                road_obj["road_name"] = road["name"]
                roads.append(road_obj)

        self.logger.info(f"✅ {len(roads)} Straßen erstellt")
        return roads

    def create_buildings(
        self, osm_data: Dict[str, Any], center_lat: float, center_lon: float
    ) -> List[bpy.types.Object]:
        """Erstelle Gebäude aus OSM-Daten"""
        buildings = []

        for building in osm_data.get("buildings", []):
            if len(building["nodes"]) < 3:
                continue

            # Gebäude-Typ bestimmen
            building_type = building["type"]
            try:
                height = float(building["height"])
            except (ValueError, TypeError):
                if building_type in ["house", "residential"]:
                    height = 3.0
                elif building_type in ["apartments", "commercial"]:
                    height = 8.0
                elif building_type == "industrial":
                    height = 6.0
                else:
                    height = 4.0

            # Gebäude erstellen
            building_obj = self._create_building_mesh(
                building["nodes"], height, center_lat, center_lon
            )
            if building_obj:
                building_obj.name = f"Building_{building_type}_{building['id']}"
                building_obj["building_type"] = building_type
                building_obj["height"] = height
                buildings.append(building_obj)

        self.logger.info(f"✅ {len(buildings)} Gebäude erstellt")
        return buildings

    def create_vegetation(
        self, osm_data: Dict[str, Any], center_lat: float, center_lon: float
    ) -> List[bpy.types.Object]:
        """Erstelle Vegetation aus OSM-Daten"""
        vegetation = []

        for area in osm_data.get("natural", []):
            if area["type"] in ["forest", "wood"] and len(area["nodes"]) >= 3:
                # Wald-Gebiet
                forest_obj = self._create_forest_area(
                    area["nodes"], center_lat, center_lon
                )
                if forest_obj:
                    forest_obj.name = f"Forest_{area['id']}"
                    forest_obj["vegetation_type"] = "forest"
                    vegetation.append(forest_obj)

        for area in osm_data.get("landuse", []):
            if area["type"] in ["park", "meadow"] and len(area["nodes"]) >= 3:
                # Park-Gebiet
                park_obj = self._create_park_area(area["nodes"], center_lat, center_lon)
                if park_obj:
                    park_obj.name = f"Park_{area['id']}"
                    park_obj["vegetation_type"] = "park"
                    vegetation.append(park_obj)

        self.logger.info(f"✅ {len(vegetation)} Vegetations-Gebiete erstellt")
        return vegetation

    def _create_road_mesh(
        self,
        nodes: List[Tuple[float, float]],
        width: float,
        height: float,
        center_lat: float,
        center_lon: float,
    ) -> Optional[bpy.types.Object]:
        """Erstelle Straßen-Mesh"""
        try:
            # Koordinaten konvertieren
            blender_coords = []
            for lat, lon in nodes:
                x, y = self.latlon_to_blender(lat, lon, center_lat, center_lon)
                blender_coords.append((x, y, 0))

            if len(blender_coords) < 2:
                return None

            # Straßen-Mesh erstellen
            mesh = bpy.data.meshes.new(f"Road_Mesh")
            obj = bpy.data.objects.new(f"Road", mesh)
            bpy.context.collection.objects.link(obj)

            # Vereinfachte Straßen-Erstellung
            vertices = []
            faces = []

            for i in range(len(blender_coords) - 1):
                p1 = np.array(blender_coords[i])
                p2 = np.array(blender_coords[i + 1])

                # Richtungsvektor
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction)

                # Senkrechter Vektor
                perpendicular = np.array([-direction[1], direction[0], 0])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)

                # Straßen-Ränder
                offset = perpendicular * width / 2

                # Vertices hinzufügen
                v1 = p1 + offset
                v2 = p1 - offset
                v3 = p2 + offset
                v4 = p2 - offset

                start_idx = len(vertices)
                vertices.extend([v1, v2, v3, v4])

                # Faces hinzufügen
                faces.append([start_idx, start_idx + 1, start_idx + 3, start_idx + 2])

            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            return obj

        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Straße: {e}")
            return None

    def _create_building_mesh(
        self,
        nodes: List[Tuple[float, float]],
        height: float,
        center_lat: float,
        center_lon: float,
    ) -> Optional[bpy.types.Object]:
        """Erstelle Gebäude-Mesh"""
        try:
            # Koordinaten konvertieren
            blender_coords = []
            for lat, lon in nodes:
                x, y = self.latlon_to_blender(lat, lon, center_lat, center_lon)
                blender_coords.append((x, y, 0))

            if len(blender_coords) < 3:
                return None

            # Gebäude-Mesh erstellen
            mesh = bpy.data.meshes.new(f"Building_Mesh")
            obj = bpy.data.objects.new(f"Building", mesh)
            bpy.context.collection.objects.link(obj)

            # Boden-Vertices
            ground_vertices = [(x, y, 0) for x, y, z in blender_coords]
            # Dach-Vertices
            roof_vertices = [(x, y, height) for x, y, z in blender_coords]

            vertices = ground_vertices + roof_vertices
            faces = []

            # Boden-Face
            if len(ground_vertices) >= 3:
                faces.append(list(range(len(ground_vertices))))

            # Dach-Face (umgekehrt)
            if len(roof_vertices) >= 3:
                roof_face = list(
                    range(
                        len(ground_vertices), len(ground_vertices) + len(roof_vertices)
                    )
                )
                faces.append(roof_face[::-1])

            # Seiten-Faces
            for i in range(len(ground_vertices)):
                next_i = (i + 1) % len(ground_vertices)
                faces.append(
                    [i, next_i, len(ground_vertices) + next_i, len(ground_vertices) + i]
                )

            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            return obj

        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Gebäudes: {e}")
            return None

    def _create_forest_area(
        self, nodes: List[Tuple[float, float]], center_lat: float, center_lon: float
    ) -> Optional[bpy.types.Object]:
        """Erstelle Wald-Gebiet"""
        # Vereinfachte Implementierung - könnte mit TreeFactory erweitert werden
        return self._create_park_area(nodes, center_lat, center_lon)

    def _create_park_area(
        self, nodes: List[Tuple[float, float]], center_lat: float, center_lon: float
    ) -> Optional[bpy.types.Object]:
        """Erstelle Park-Gebiet"""
        try:
            # Koordinaten konvertieren
            blender_coords = []
            for lat, lon in nodes:
                x, y = self.latlon_to_blender(lat, lon, center_lat, center_lon)
                blender_coords.append((x, y, 0))

            if len(blender_coords) < 3:
                return None

            # Park-Mesh erstellen
            mesh = bpy.data.meshes.new(f"Park_Mesh")
            obj = bpy.data.objects.new(f"Park", mesh)
            bpy.context.collection.objects.link(obj)

            vertices = [(x, y, 0) for x, y, z in blender_coords]
            faces = [list(range(len(vertices)))]

            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            return obj

        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Parks: {e}")
            return None


class InfinigenOSMIntegrator:
    """Hauptklasse für OSM-Integration in Infinigen"""

    def __init__(self):
        self.fetcher = OSMDataFetcher()
        self.converter = OSMToBlenderConverter()
        self.logger = logging.getLogger(__name__)

    def generate_urban_scene(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float = 1.0,
        scene_seed: int = 42,
    ) -> Dict[str, Any]:
        """Generiere städtische Szene mit OSM-Daten"""

        # Bounding Box berechnen
        lat_offset = radius_km / 111.0  # Grob 1km = 1/111 Grad
        lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        bbox = (
            center_lat - lat_offset,  # min_lat
            center_lon - lon_offset,  # min_lon
            center_lat + lat_offset,  # max_lat
            center_lon + lon_offset,  # max_lon
        )

        self.logger.info(
            f"Generiere städtische Szene für {center_lat}, {center_lon} (Radius: {radius_km}km)"
        )

        # OSM-Daten laden
        osm_data = self.fetcher.fetch_area_data(bbox, "urban")
        if "error" in osm_data:
            return {"success": False, "error": osm_data["error"]}

        # Blender-Objekte erstellen
        roads = self.converter.create_roads(osm_data, center_lat, center_lon)
        buildings = self.converter.create_buildings(osm_data, center_lat, center_lon)
        vegetation = self.converter.create_vegetation(osm_data, center_lat, center_lon)

        # Sammlung erstellen
        collection = bpy.data.collections.new(f"OSM_Scene_{scene_seed}")
        bpy.context.scene.collection.children.link(collection)

        # Objekte zur Sammlung hinzufügen
        for obj in roads + buildings + vegetation:
            collection.objects.link(obj)

        return {
            "success": True,
            "roads_count": len(roads),
            "buildings_count": len(buildings),
            "vegetation_count": len(vegetation),
            "collection": collection,
            "osm_data": osm_data,
        }

    def generate_rural_scene(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float = 2.0,
        scene_seed: int = 42,
    ) -> Dict[str, Any]:
        """Generiere ländliche Szene mit OSM-Daten"""

        # Bounding Box berechnen
        lat_offset = radius_km / 111.0
        lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        bbox = (
            center_lat - lat_offset,
            center_lon - lon_offset,
            center_lat + lat_offset,
            center_lon + lon_offset,
        )

        self.logger.info(
            f"Generiere ländliche Szene für {center_lat}, {center_lon} (Radius: {radius_km}km)"
        )

        # OSM-Daten laden
        osm_data = self.fetcher.fetch_area_data(bbox, "rural")
        if "error" in osm_data:
            return {"success": False, "error": osm_data["error"]}

        # Blender-Objekte erstellen
        roads = self.converter.create_roads(osm_data, center_lat, center_lon)
        buildings = self.converter.create_buildings(osm_data, center_lat, center_lon)
        vegetation = self.converter.create_vegetation(osm_data, center_lat, center_lon)

        # Sammlung erstellen
        collection = bpy.data.collections.new(f"OSM_Rural_Scene_{scene_seed}")
        bpy.context.scene.collection.children.link(collection)

        # Objekte zur Sammlung hinzufügen
        for obj in roads + buildings + vegetation:
            collection.objects.link(obj)

        return {
            "success": True,
            "roads_count": len(roads),
            "buildings_count": len(buildings),
            "vegetation_count": len(vegetation),
            "collection": collection,
            "osm_data": osm_data,
        }


# Dependency für Agents
def get_osm_integrator() -> InfinigenOSMIntegrator:
    """Hole OSM-Integrator für Agent"""
    return InfinigenOSMIntegrator()
