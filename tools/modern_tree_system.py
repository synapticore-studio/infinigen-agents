#!/usr/bin/env python3
"""
Modernes Tree-System für Infinigen
Ersetzt das alte, komplexe Tree-System durch moderne Addons
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bmesh
import bpy
import numpy as np

logger = logging.getLogger(__name__)


class ModernTreeGenerator:
    """Moderne Baum-Generierung ohne komplexe Nodes"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Baum-Typen und Parameter
        self.tree_types = {
            "oak": {
                "trunk_height": (3.0, 8.0),
                "trunk_radius": (0.3, 0.8),
                "crown_radius": (2.0, 5.0),
                "crown_height": (2.0, 4.0),
                "branch_count": (8, 15),
                "leaf_density": 0.7,
            },
            "pine": {
                "trunk_height": (5.0, 15.0),
                "trunk_radius": (0.2, 0.6),
                "crown_radius": (1.5, 3.0),
                "crown_height": (3.0, 8.0),
                "branch_count": (12, 20),
                "leaf_density": 0.9,
            },
            "maple": {
                "trunk_height": (4.0, 10.0),
                "trunk_radius": (0.4, 0.9),
                "crown_radius": (3.0, 6.0),
                "crown_height": (2.5, 5.0),
                "branch_count": (6, 12),
                "leaf_density": 0.8,
            },
            "palm": {
                "trunk_height": (6.0, 12.0),
                "trunk_radius": (0.3, 0.7),
                "crown_radius": (2.0, 4.0),
                "crown_height": (1.0, 2.0),
                "branch_count": (8, 15),
                "leaf_density": 0.6,
            },
            "bush": {
                "trunk_height": (0.5, 2.0),
                "trunk_radius": (0.1, 0.3),
                "crown_radius": (1.0, 3.0),
                "crown_height": (1.0, 2.5),
                "branch_count": (4, 8),
                "leaf_density": 0.9,
            },
        }

    def generate_tree(
        self,
        tree_type: str = "oak",
        seed: int = 42,
        position: Tuple[float, float, float] = (0, 0, 0),
        scale: float = 1.0,
    ) -> Optional[bpy.types.Object]:
        """Generiere einen modernen Baum"""

        if tree_type not in self.tree_types:
            tree_type = "oak"

        # Seed setzen für reproduzierbare Ergebnisse
        np.random.seed(seed)

        try:
            # Baum-Parameter
            params = self.tree_types[tree_type]

            # Zufällige Parameter basierend auf Seed
            trunk_height = np.random.uniform(*params["trunk_height"]) * scale
            trunk_radius = np.random.uniform(*params["trunk_radius"]) * scale
            crown_radius = np.random.uniform(*params["crown_radius"]) * scale
            crown_height = np.random.uniform(*params["crown_height"]) * scale
            branch_count = np.random.randint(*params["branch_count"])
            leaf_density = params["leaf_density"]

            # Baum erstellen
            tree_obj = self._create_tree_mesh(
                tree_type,
                trunk_height,
                trunk_radius,
                crown_radius,
                crown_height,
                branch_count,
                leaf_density,
                position,
            )

            if tree_obj:
                tree_obj.name = f"{tree_type.title()}_Tree_{seed}"
                tree_obj["tree_type"] = tree_type
                tree_obj["tree_seed"] = seed
                tree_obj["generation_time"] = time.time()

                # Material anwenden
                self._apply_tree_material(tree_obj, tree_type)

                self.logger.info(
                    f"✅ {tree_type.title()}-Baum generiert: {tree_obj.name}"
                )
                return tree_obj

        except Exception as e:
            self.logger.error(f"❌ Fehler beim Generieren des Baums: {e}")
            return None

    def _create_tree_mesh(
        self,
        tree_type: str,
        trunk_height: float,
        trunk_radius: float,
        crown_radius: float,
        crown_height: float,
        branch_count: int,
        leaf_density: float,
        position: Tuple[float, float, float],
    ) -> Optional[bpy.types.Object]:
        """Erstelle Baum-Mesh"""

        try:
            # Neues Mesh erstellen
            mesh = bpy.data.meshes.new(f"{tree_type}_Tree_Mesh")
            obj = bpy.data.objects.new(f"{tree_type}_Tree", mesh)
            bpy.context.collection.objects.link(obj)

            # Position setzen
            obj.location = position

            # Bmesh für Mesh-Erstellung
            bm = bmesh.new()

            if tree_type == "palm":
                # Palme - spezielle Form
                self._create_palm_mesh(
                    bm, trunk_height, trunk_radius, crown_radius, crown_height
                )
            else:
                # Standard-Baum
                self._create_standard_tree_mesh(
                    bm,
                    trunk_height,
                    trunk_radius,
                    crown_radius,
                    crown_height,
                    branch_count,
                )

            # Mesh finalisieren
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

            return obj

        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Baum-Meshes: {e}")
            return None

    def _create_standard_tree_mesh(
        self,
        bm: bmesh.types.BMesh,
        trunk_height: float,
        trunk_radius: float,
        crown_radius: float,
        crown_height: float,
        branch_count: int,
    ):
        """Erstelle Standard-Baum (Eiche, Kiefer, Ahorn)"""

        # Stamm erstellen
        trunk_cylinder = bmesh.ops.create_cylinder(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=8,
            radius=trunk_radius,
            depth=trunk_height,
        )

        # Stamm nach oben verschieben
        bmesh.ops.translate(
            bm, vec=(0, 0, trunk_height / 2), verts=trunk_cylinder["verts"]
        )

        # Krone erstellen
        crown_sphere = bmesh.ops.create_uv_sphere(
            bm,
            u_segments=8,
            v_segments=6,
            radius=crown_radius,
            center=(0, 0, trunk_height + crown_height / 2),
        )

        # Krone skalieren (elliptisch)
        bmesh.ops.scale(
            bm, vec=(1.0, 1.0, crown_height / crown_radius), verts=crown_sphere["verts"]
        )

        # Äste hinzufügen (vereinfacht)
        for i in range(branch_count):
            angle = (2 * np.pi * i) / branch_count
            branch_height = trunk_height * (0.3 + 0.4 * np.random.random())
            branch_length = crown_radius * (0.5 + 0.5 * np.random.random())

            # Ast-Position
            x = np.cos(angle) * trunk_radius * 1.2
            y = np.sin(angle) * trunk_radius * 1.2
            z = branch_height

            # Ast erstellen
            branch_cylinder = bmesh.ops.create_cylinder(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=6,
                radius=trunk_radius * 0.3,
                depth=branch_length,
            )

            # Ast positionieren und rotieren
            bmesh.ops.translate(bm, vec=(x, y, z), verts=branch_cylinder["verts"])

            # Ast rotieren
            rotation_angle = np.random.uniform(-30, 30)  # Grad
            bmesh.ops.rotate(
                bm,
                cent=(x, y, z),
                matrix=np.array(
                    [
                        [
                            np.cos(np.radians(rotation_angle)),
                            -np.sin(np.radians(rotation_angle)),
                            0,
                        ],
                        [
                            np.sin(np.radians(rotation_angle)),
                            np.cos(np.radians(rotation_angle)),
                            0,
                        ],
                        [0, 0, 1],
                    ]
                ),
                verts=branch_cylinder["verts"],
            )

    def _create_palm_mesh(
        self,
        bm: bmesh.types.BMesh,
        trunk_height: float,
        trunk_radius: float,
        crown_radius: float,
        crown_height: float,
    ):
        """Erstelle Palmen-Mesh"""

        # Stamm erstellen (schlanker)
        trunk_cylinder = bmesh.ops.create_cylinder(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=8,
            radius=trunk_radius,
            depth=trunk_height,
        )

        # Stamm nach oben verschieben
        bmesh.ops.translate(
            bm, vec=(0, 0, trunk_height / 2), verts=trunk_cylinder["verts"]
        )

        # Palmenwedel erstellen
        for i in range(8):  # 8 Wedel
            angle = (2 * np.pi * i) / 8
            wedge_length = crown_radius * (0.8 + 0.4 * np.random.random())

            # Wedel-Position
            x = np.cos(angle) * trunk_radius * 1.1
            y = np.sin(angle) * trunk_radius * 1.1
            z = trunk_height

            # Wedel erstellen (vereinfacht als Kegel)
            wedge_cone = bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                segments=6,
                radius1=wedge_length * 0.3,
                radius2=0.1,
                depth=wedge_length,
            )

            # Wedel positionieren
            bmesh.ops.translate(bm, vec=(x, y, z), verts=wedge_cone["verts"])

            # Wedel rotieren
            rotation_angle = angle * 180 / np.pi
            bmesh.ops.rotate(
                bm,
                cent=(x, y, z),
                matrix=np.array(
                    [
                        [
                            np.cos(np.radians(rotation_angle)),
                            -np.sin(np.radians(rotation_angle)),
                            0,
                        ],
                        [
                            np.sin(np.radians(rotation_angle)),
                            np.cos(np.radians(rotation_angle)),
                            0,
                        ],
                        [0, 0, 1],
                    ]
                ),
                verts=wedge_cone["verts"],
            )

    def _apply_tree_material(self, obj: bpy.types.Object, tree_type: str):
        """Wende Baum-Material an"""

        try:
            # Material erstellen
            material = bpy.data.materials.new(name=f"{tree_type.title()}_Material")
            material.use_nodes = True

            # Principled BSDF konfigurieren
            bsdf = material.node_tree.nodes["Principled BSDF"]

            if tree_type == "oak":
                bsdf.inputs["Base Color"].default_value = (0.4, 0.3, 0.1, 1.0)  # Braun
                bsdf.inputs["Roughness"].default_value = 0.8
            elif tree_type == "pine":
                bsdf.inputs["Base Color"].default_value = (0.2, 0.4, 0.2, 1.0)  # Grün
                bsdf.inputs["Roughness"].default_value = 0.9
            elif tree_type == "maple":
                bsdf.inputs["Base Color"].default_value = (
                    0.6,
                    0.4,
                    0.2,
                    1.0,
                )  # Orange-Braun
                bsdf.inputs["Roughness"].default_value = 0.7
            elif tree_type == "palm":
                bsdf.inputs["Base Color"].default_value = (0.3, 0.5, 0.2, 1.0)  # Grün
                bsdf.inputs["Roughness"].default_value = 0.6
            else:  # bush
                bsdf.inputs["Base Color"].default_value = (0.2, 0.6, 0.2, 1.0)  # Grün
                bsdf.inputs["Roughness"].default_value = 0.8

            # Material zu Objekt hinzufügen
            obj.data.materials.append(material)

        except Exception as e:
            self.logger.warning(f"Material konnte nicht angewendet werden: {e}")

    def generate_forest(
        self,
        tree_count: int = 50,
        area_size: float = 100.0,
        tree_types: List[str] = None,
        seed: int = 42,
    ) -> List[bpy.types.Object]:
        """Generiere einen Wald"""

        if tree_types is None:
            tree_types = ["oak", "pine", "maple"]

        np.random.seed(seed)
        trees = []

        for i in range(tree_count):
            # Zufällige Position
            x = np.random.uniform(-area_size / 2, area_size / 2)
            y = np.random.uniform(-area_size / 2, area_size / 2)
            z = 0  # Auf Terrain-Oberfläche

            # Zufälliger Baum-Typ
            tree_type = np.random.choice(tree_types)

            # Zufällige Skalierung
            scale = np.random.uniform(0.7, 1.3)

            # Baum generieren
            tree = self.generate_tree(
                tree_type=tree_type, seed=seed + i, position=(x, y, z), scale=scale
            )

            if tree:
                trees.append(tree)

        self.logger.info(f"✅ Wald generiert: {len(trees)} Bäume")
        return trees

    def get_available_tree_types(self) -> List[str]:
        """Hole verfügbare Baum-Typen"""
        return list(self.tree_types.keys())


class ModernTreeFactory:
    """Factory für moderne Bäume - ersetzt TreeFactory"""

    def __init__(self, seed: int, coarse: bool = False):
        self.seed = seed
        self.coarse = coarse
        self.generator = ModernTreeGenerator()
        self.logger = logging.getLogger(__name__)

    def create_asset(
        self, params: Dict[str, Any] = None, **kwargs
    ) -> Optional[bpy.types.Object]:
        """Erstelle Baum-Asset"""

        if params is None:
            params = {}

        # Parameter extrahieren
        tree_type = params.get("tree_type", "oak")
        position = params.get("position", (0, 0, 0))
        scale = params.get("scale", 1.0)

        # Baum generieren
        tree = self.generator.generate_tree(
            tree_type=tree_type, seed=self.seed, position=position, scale=scale
        )

        return tree


# Dependency für Agents
def get_modern_tree_generator() -> ModernTreeGenerator:
    """Hole modernen Tree-Generator"""
    return ModernTreeGenerator()


def get_modern_tree_factory(seed: int, coarse: bool = False) -> ModernTreeFactory:
    """Hole moderne Tree-Factory"""
    return ModernTreeFactory(seed, coarse)
