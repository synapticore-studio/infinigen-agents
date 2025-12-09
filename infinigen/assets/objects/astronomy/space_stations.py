# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: AI Assistant
# Astronomical Space Stations and Satellites for Infinigen

import bpy
import gin
import numpy as np
from mathutils import Vector
from numpy.random import normal, uniform

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import SurfaceTypes


@gin.configurable
class SpaceStationFactory(AssetFactory):
    """Space station factory for different types of stations"""

    def __init__(self, factory_seed, station_type="modular", size=5.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.station_type = station_type
        self.size = size

    def create_space_station_geometry(self, nw: NodeWrangler):
        """Create space station geometry based on type"""
        group_input = nw.new_node(Nodes.GroupInput)

        if self.station_type == "modular":
            return self.create_modular_station(nw)
        elif self.station_type == "ring":
            return self.create_ring_station(nw)
        elif self.station_type == "cylindrical":
            return self.create_cylindrical_station(nw)
        else:
            return self.create_modular_station(nw)

    def create_modular_station(self, nw: NodeWrangler):
        """Create modular space station with multiple components"""
        # Create central hub using cube
        central_hub = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (4.0, 4.0, 2.0)}
        )

        # Create modular arms using cubes
        arm_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.6, 8.0, 0.6)})

        # Position arm 1
        arm_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": arm_1, "Offset": (4.0, 0.0, 0.0)},
        )

        # Create second arm
        arm_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.6, 8.0, 0.6)})

        # Position arm 2
        arm_2_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": arm_2, "Offset": (-4.0, 0.0, 0.0)},
        )

        # Create docking modules
        dock_module_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (2.0, 2.0, 2.0)}
        )

        # Position docking module 1
        dock_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": dock_module_1, "Offset": (8.0, 0.0, 0.0)},
        )

        # Create solar panels
        solar_panel = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.1, 6.0, 4.0)}
        )

        # Position solar panels
        solar_1 = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel, "Offset": (0.0, 3.0, 0.0)},
        )

        solar_2 = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel, "Offset": (0.0, -3.0, 0.0)},
        )

        # Join all components
        station_geometry = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    central_hub,
                    arm_1_positioned,
                    arm_2_positioned,
                    dock_1_positioned,
                    solar_1,
                    solar_2,
                ]
            },
        )

        return station_geometry

    def create_ring_station(self, nw: NodeWrangler):
        """Create ring-shaped space station"""
        # Create main ring using cube
        ring = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (10.0, 10.0, 1.0)})

        # Create central hub
        central_hub = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (2.0, 2.0, 2.0)}
        )

        # Create connecting spokes
        spoke_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.2, 5.0, 0.2)})

        spoke_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": spoke_1, "Offset": (2.5, 0.0, 0.0)},
        )

        # Join components
        ring_station = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={"Geometry": [ring, central_hub, spoke_1_positioned]},
        )

        return ring_station

    def create_cylindrical_station(self, nw: NodeWrangler):
        """Create cylindrical space station"""
        # Create main cylinder using cube
        main_cylinder = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (6.0, 6.0, 8.0)}
        )

        # Create end caps
        end_cap_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (6.0, 6.0, 0.5)})

        end_cap_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": end_cap_1, "Offset": (0.0, 0.0, 4.25)},
        )

        end_cap_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (6.0, 6.0, 0.5)})

        end_cap_2_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": end_cap_2, "Offset": (0.0, 0.0, -4.25)},
        )

        # Create docking ports
        dock_port = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.0, 1.0, 1.0)})

        dock_port_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": dock_port, "Offset": (3.5, 0.0, 0.0)},
        )

        # Join components
        cylindrical_station = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    main_cylinder,
                    end_cap_1_positioned,
                    end_cap_2_positioned,
                    dock_port_positioned,
                ]
            },
        )

        return cylindrical_station

    def create_asset(self, **kwargs):
        """Create space station asset"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=self.size, location=(0, 0, 0))
        station = bpy.context.active_object
        station.name = f"SpaceStation_{self.factory_seed}"

        # Add geometry nodes for space station
        surface.add_geomod(station, self.create_space_station_geometry, apply=True)

        # Tag object
        tag_object(station, SurfaceTypes.SDFPerturb)

        return station


@gin.configurable
class SatelliteFactory(AssetFactory):
    """Satellite factory for different types of satellites"""

    def __init__(
        self, factory_seed, satellite_type="communication", size=1.0, coarse=False
    ):
        super().__init__(factory_seed, coarse=coarse)
        self.satellite_type = satellite_type
        self.size = size

    def create_satellite_geometry(self, nw: NodeWrangler):
        """Create satellite geometry with solar panels and antennas"""
        group_input = nw.new_node(Nodes.GroupInput)

        if self.satellite_type == "communication":
            return self.create_communication_satellite(nw)
        elif self.satellite_type == "weather":
            return self.create_weather_satellite(nw)
        elif self.satellite_type == "research":
            return self.create_research_satellite(nw)
        else:
            return self.create_communication_satellite(nw)

    def create_communication_satellite(self, nw: NodeWrangler):
        """Create communication satellite with large solar panels and antennas"""
        # Main satellite body
        main_body = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.0, 1.0, 0.5)})

        # Solar panels
        solar_panel_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.05, 3.0, 2.0)}
        )

        solar_panel_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel_1, "Offset": (0.0, 1.5, 0.0)},
        )

        solar_panel_2 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.05, 3.0, 2.0)}
        )

        solar_panel_2_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel_2, "Offset": (0.0, -1.5, 0.0)},
        )

        # Communication dish
        dish = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.6, 1.6, 0.1)})

        dish_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": dish, "Offset": (0.0, 0.0, 0.3)},
        )

        # Antennas
        antenna_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.04, 0.04, 1.0)}
        )

        antenna_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": antenna_1, "Offset": (0.0, 0.0, 0.8)},
        )

        # Join components
        satellite = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    main_body,
                    solar_panel_1_positioned,
                    solar_panel_2_positioned,
                    dish_positioned,
                    antenna_1_positioned,
                ]
            },
        )

        return satellite

    def create_weather_satellite(self, nw: NodeWrangler):
        """Create weather satellite with sensors and instruments"""
        # Main body
        main_body = nw.new_node(
            Nodes.MeshCylinder,
            input_kwargs={"Radius": 0.5, "Depth": 0.3, "Vertices": 12},
        )

        # Solar panel array
        solar_array = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.05, 2.0, 1.5)}
        )

        solar_array_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_array, "Offset": (0.0, 1.0, 0.0)},
        )

        # Weather instruments
        instrument_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.2, 0.2, 0.2)}
        )

        instrument_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": instrument_1, "Offset": (0.0, 0.0, 0.3)},
        )

        # Join components
        weather_satellite = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [main_body, solar_array_positioned, instrument_1_positioned]
            },
        )

        return weather_satellite

    def create_research_satellite(self, nw: NodeWrangler):
        """Create research satellite with multiple instruments"""
        # Main body
        main_body = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.8, 0.8, 0.4)})

        # Solar panels
        solar_panel = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.05, 2.5, 1.0)}
        )

        solar_panel_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel, "Offset": (0.0, 1.25, 0.0)},
        )

        # Research instruments
        instrument_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.3, 0.3, 0.3)}
        )

        instrument_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": instrument_1, "Offset": (0.0, 0.0, 0.4)},
        )

        # Join components
        research_satellite = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [main_body, solar_panel_positioned, instrument_1_positioned]
            },
        )

        return research_satellite

    def create_asset(self, **kwargs):
        """Create satellite asset"""
        bpy.ops.mesh.primitive_cube_add(size=self.size, location=(0, 0, 0))
        satellite = bpy.context.active_object
        satellite.name = f"Satellite_{self.factory_seed}"

        # Add geometry nodes for satellite
        surface.add_geomod(satellite, self.create_satellite_geometry, apply=True)

        # Tag object
        tag_object(satellite, SurfaceTypes.SDFPerturb)

        return satellite


@gin.configurable
class SpacecraftFactory(AssetFactory):
    """Spacecraft factory for different types of spacecraft"""

    def __init__(self, factory_seed, spacecraft_type="shuttle", size=3.0, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        self.spacecraft_type = spacecraft_type
        self.size = size

    def create_spacecraft_geometry(self, nw: NodeWrangler):
        """Create spacecraft geometry based on type"""
        group_input = nw.new_node(Nodes.GroupInput)

        if self.spacecraft_type == "shuttle":
            return self.create_space_shuttle(nw)
        elif self.spacecraft_type == "capsule":
            return self.create_capsule(nw)
        elif self.spacecraft_type == "lander":
            return self.create_lander(nw)
        else:
            return self.create_space_shuttle(nw)

    def create_space_shuttle(self, nw: NodeWrangler):
        """Create space shuttle with wings and engines"""
        # Main fuselage
        fuselage = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.6, 0.6, 3.0)})

        # Wings
        wing_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.1, 1.5, 0.8)})

        wing_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": wing_1, "Offset": (0.0, 0.75, 0.0)},
        )

        wing_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.1, 1.5, 0.8)})

        wing_2_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": wing_2, "Offset": (0.0, -0.75, 0.0)},
        )

        # Engines
        engine_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.2, 0.2, 0.5)})

        engine_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": engine_1, "Offset": (0.0, 0.0, -1.75)},
        )

        # Join components
        shuttle = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    fuselage,
                    wing_1_positioned,
                    wing_2_positioned,
                    engine_1_positioned,
                ]
            },
        )

        return shuttle

    def create_capsule(self, nw: NodeWrangler):
        """Create capsule spacecraft"""
        # Main capsule body
        capsule = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.8, 0.8, 1.0)})

        # Heat shield
        heat_shield = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.9, 0.9, 0.1)}
        )

        heat_shield_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": heat_shield, "Offset": (0.0, 0.0, -0.55)},
        )

        # Parachute compartment
        parachute_compartment = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.6, 0.6, 0.2)}
        )

        parachute_compartment_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": parachute_compartment, "Offset": (0.0, 0.0, 0.6)},
        )

        # Join components
        capsule_spacecraft = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    capsule,
                    heat_shield_positioned,
                    parachute_compartment_positioned,
                ]
            },
        )

        return capsule_spacecraft

    def create_lander(self, nw: NodeWrangler):
        """Create planetary lander"""
        # Main body
        main_body = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (1.0, 1.0, 0.3)})

        # Landing legs
        leg_1 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.1, 0.1, 0.8)})

        leg_1_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": leg_1, "Offset": (0.4, 0.0, -0.4)},
        )

        leg_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": (0.1, 0.1, 0.8)})

        leg_2_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": leg_2, "Offset": (-0.4, 0.0, -0.4)},
        )

        # Solar panels
        solar_panel = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.05, 1.0, 0.5)}
        )

        solar_panel_positioned = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={"Geometry": solar_panel, "Offset": (0.0, 0.5, 0.0)},
        )

        # Join components
        lander = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    main_body,
                    leg_1_positioned,
                    leg_2_positioned,
                    solar_panel_positioned,
                ]
            },
        )

        return lander

    def create_asset(self, **kwargs):
        """Create spacecraft asset"""
        bpy.ops.mesh.primitive_cone_add(
            radius1=self.size, depth=self.size * 2, location=(0, 0, 0)
        )
        spacecraft = bpy.context.active_object
        spacecraft.name = f"Spacecraft_{self.factory_seed}"

        # Add geometry nodes for spacecraft
        surface.add_geomod(spacecraft, self.create_spacecraft_geometry, apply=True)

        # Tag object
        tag_object(spacecraft, SurfaceTypes.SDFPerturb)

        return spacecraft
