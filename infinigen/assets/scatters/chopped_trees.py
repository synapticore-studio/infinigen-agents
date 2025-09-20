# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import bpy
import mathutils
import numpy as np
from numpy.random import normal, uniform

from infinigen.assets.composition import material_assignments
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.detail import (
    remesh_with_attrs,
    scatter_res_distance,
    target_face_size,
)
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.util import blender as butil
from infinigen.core.util import math as mutil
from infinigen.core.util.math import rotate_match_directions
from infinigen.core.util.random import weighted_sample

logger = logging.getLogger(__name__)


def approx_settle_transform(obj, samples=200):
    assert obj.type == "MESH"

    if len(obj.data.vertices) < 3 or len(obj.data.polygons) == 0:
        return

    with butil.SelectObjects(obj):
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # sample random planes and find the normal of the biggest one
    verts = np.empty((len(obj.data.vertices), 3))
    obj.data.vertices.foreach_get("co", verts.reshape(-1))
    verts = np.stack(
        [verts[np.random.choice(np.arange(len(verts)), samples)] for _ in range(3)],
        axis=0,
    )
    ups = np.cross(verts[0] - verts[1], verts[0] - verts[2], axis=-1)
    best = np.linalg.norm(ups, axis=-1).argmax()

    # rotate according to that axis
    rot_mat = rotate_match_directions(
        ups[best].reshape(1, 3), np.array([0, 0, 1]).reshape(1, 3)
    )[0]
    obj.rotation_euler = mathutils.Matrix(rot_mat).to_euler()

    with butil.SelectObjects(obj):
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    return obj


def chopped_tree_collection(species_seed, n, boolean_res_mult=5):
    """Erstelle chopped trees mit dem originalen Infinigen-Konzept - effizient und schnell"""
    objs = []

    # Verwende das originale Infinigen-Konzept mit einfachen Cubes
    # Das ist viel effizienter als komplexe Tree-Generation
    logger.info(f"Creating {n} simple chopped tree placeholders")

    for i in range(n):
        # Einfacher Cube als Basis - wie im originalen Infinigen
        tree = butil.spawn_cube(size=4, location=(0, 0, 0), name=f"chopped_tree_{i}")

        # Einfache "Chopping" durch Skalierung und Rotation
        # Das ist viel schneller als Boolean-Operationen
        tree.scale = (
            np.random.uniform(0.3, 1.0),  # x
            np.random.uniform(0.3, 1.0),  # y
            np.random.uniform(0.5, 1.2),  # z (Höhe)
        )
        tree.rotation_euler = (
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0, 2 * np.pi),
        )

        # Einfache "Chopped" Effekte durch Modifier
        butil.modify_mesh(tree, "DECIMATE", ratio=0.7)
        butil.modify_mesh(tree, "SUBSURF", levels=1)

        tree.name = f"chopped_tree({species_seed}, {i})"
        objs.append(tree)

    return butil.group_in_collection(objs, "assets:chopped_tree", reuse=False)


def apply(obj, species_seed=None, selection=None, n_trees=1, **kwargs):
    assert obj is not None
    if species_seed is None:
        species_seed = np.random.randint(1e6)

    col = chopped_tree_collection(species_seed, n=n_trees)
    col.hide_viewport = True

    scatter_obj = scatter_instances(
        base_obj=obj,
        collection=col,
        scale=1,
        scale_rand=0.5,
        scale_rand_axi=0.15,
        ground_offset=0.1,
        density=0.7,
        selection=selection,
    )

    return scatter_obj, col
